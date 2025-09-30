/**
 * offer_curation.ts
 *
 * A TypeScript translation of the original Python notebook used for grading Klook offers.
 * The script loads offer JSON files, prepares payloads for the OpenAI Responses API,
 * evaluates each offer, logs the graded responses, and exports the aggregated results to CSV.
 */

// --- Section: Standard library imports and environment bootstrap ---
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { OpenAI } from 'openai';
import dotenv from 'dotenv';

// Resolve the current directory when running under ESM-compatible loaders.
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

// --- Section: Shared constants mirroring the Python notebook configuration ---
const OFFERS_DIR = path.resolve(__dirname, 'offers');
const MAX_IMAGES_TO_REVIEW = 8;
const MODEL_NAME = 'gpt-5';
const REASONING_EFFORT = 'medium';
const MAX_OUTPUT_TOKENS = 5000;
const OPENAI_API_KEY_ENV = 'OPENAI_API_KEY';
const ENV_PRIORITIES = ['.env', '.openai_api_key'];
const CSV_OUTPUT_PATH = path.resolve(__dirname, 'graded_offers_ts.csv');
const CONCURRENCY = Number.parseInt(process.env.OFFER_GRADING_CONCURRENCY || '4', 10);

// --- Section: TypeScript type helpers describing the offer payloads ---
type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

interface ImageDetail {
  url?: string;
  type?: string;
  alt?: string;
  description?: string;
  width?: number;
  height?: number;
  source?: string;
}

interface SectionGroup {
  group_name?: string;
  content?: string;
}

interface SectionInfo {
  section_name?: string;
  groups?: SectionGroup[];
}

interface PackageSummary {
  package_id?: JsonValue;
  package_name?: string;
  sections_markdown?: string;
}

interface StructuredOffer {
  source_path: string;
  activity_id?: JsonValue;
  title?: string;
  subtitle?: string;
  what_we_love?: string;
  location?: JsonValue;
  address?: JsonValue;
  category?: string;
  category_detail?: Record<string, JsonValue> | null;
  description_markdown?: string;
  packages: PackageSummary[];
  images: string[];
  image_details: ImageDetail[];
  city?: string;
  country?: string;
  status?: JsonValue;
  raw: Record<string, JsonValue>;
}

interface GradingResult {
  activity_id: JsonValue;
  score: number | null;
  reason: string;
  categories: string[];
  target_audiences: string[];
  hero_image_index: number | null;
  hero_image_url: string | null;
  hero_image_reason: string;
  response_id: string | null;
}

// --- Section: Utility functions for environment discovery ---
function loadApiKeyFromFile(candidatePath: string): string | undefined {
  try {
    const content = fs.readFileSync(candidatePath, 'utf-8');
    for (const rawLine of content.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith('#') || !line.includes('=')) {
        continue;
      }
      const [key, value] = line.split('=', 2);
      if (key.trim() === OPENAI_API_KEY_ENV) {
        return value.trim().replace(/^"|"$/g, '');
      }
    }
  } catch (error) {
    // Silently ignore missing files; higher priority sources may exist.
  }
  return undefined;
}

function loadApiKey(): string {
  const envValue = process.env[OPENAI_API_KEY_ENV];
  if (envValue && envValue.trim()) {
    return envValue.trim();
  }

  for (const candidate of ENV_PRIORITIES) {
    const candidatePath = path.resolve(__dirname, candidate);
    const key = loadApiKeyFromFile(candidatePath);
    if (key) {
      return key;
    }
  }

  throw new Error(
    'OpenAI API key not found. Set OPENAI_API_KEY, or define it inside a local .env/.openai_api_key file.'
  );
}

// Lazily instantiate the OpenAI client so we only load credentials when we need them.
let cachedClient: OpenAI | null = null;
function getClient(): OpenAI {
  if (!cachedClient) {
    const apiKey = loadApiKey();
    cachedClient = new OpenAI({ apiKey });
  }
  return cachedClient;
}

// --- Section: JSON loading and transformation helpers ---
async function loadOffers(directory: string): Promise<StructuredOffer[]> {
  const results: StructuredOffer[] = [];
  let fileNames: string[] = [];

  try {
    fileNames = await fs.promises.readdir(directory);
  } catch (error) {
    throw new Error(`Failed to read offers directory: ${directory}`);
  }

  const jsonFiles = fileNames.filter((name) => name.toLowerCase().endsWith('.json')).sort();

  for (const fileName of jsonFiles) {
    const filePath = path.join(directory, fileName);
    try {
      const raw = await fs.promises.readFile(filePath, 'utf-8');
      const payload = JSON.parse(raw);
      const activity = payload?.activity as Record<string, JsonValue> | undefined;
      if (activity) {
        results.push(structureActivity(activity, filePath));
      }
    } catch (error) {
      console.warn(`Skipping ${filePath} due to read/parse error:`, error);
    }
  }

  return results;
}

function extractImageDetails(images: unknown): ImageDetail[] {
  const details: ImageDetail[] = [];
  if (!Array.isArray(images)) {
    return details;
  }

  for (const image of images) {
    if (!image || typeof image !== 'object') {
      continue;
    }
    const primary = image as Record<string, JsonValue>;
    const url = typeof primary.image_url_host === 'string' ? primary.image_url_host : undefined;
    if (url) {
      details.push({
        url,
        type: typeof primary.image_type === 'string' ? primary.image_type : undefined,
        alt: typeof primary.image_alt === 'string' ? primary.image_alt : undefined,
        description: typeof primary.image_desc === 'string' ? primary.image_desc : undefined,
        width: typeof primary.width === 'number' ? primary.width : undefined,
        height: typeof primary.height === 'number' ? primary.height : undefined,
        source: 'primary',
      });
    }

    const nested = primary.images;
    if (Array.isArray(nested)) {
      for (const item of nested) {
        if (!item || typeof item !== 'object') {
          continue;
        }
        const detail = item as Record<string, JsonValue>;
        const nestedUrl = typeof detail.image_url_host === 'string' ? detail.image_url_host : undefined;
        if (!nestedUrl) {
          continue;
        }
        details.push({
          url: nestedUrl,
          type: typeof detail.image_type === 'string' ? detail.image_type : undefined,
          alt: typeof detail.image_alt === 'string' ? detail.image_alt : undefined,
          description: typeof detail.image_desc === 'string' ? detail.image_desc : undefined,
          width: typeof detail.width === 'number' ? detail.width : undefined,
          height: typeof detail.height === 'number' ? detail.height : undefined,
          source: 'nested',
        });
      }
    }
  }

  const seen = new Set<string>();
  const unique: ImageDetail[] = [];
  for (const detail of details) {
    if (!detail.url || seen.has(detail.url)) {
      continue;
    }
    seen.add(detail.url);
    unique.push(detail);
  }

  return unique;
}

function renderSections(sectionInfo: unknown): string {
  if (!Array.isArray(sectionInfo)) {
    return '';
  }

  const chunks: string[] = [];

  for (const section of sectionInfo) {
    if (!section || typeof section !== 'object') {
      continue;
    }
    const block = section as SectionInfo;
    const sectionName = (block.section_name || '').toString().trim();
    const groups = Array.isArray(block.groups) ? block.groups : [];

    const groupBlocks: string[] = [];
    for (const group of groups) {
      if (!group) {
        continue;
      }
      const groupName = (group.group_name || '').toString().trim();
      const content = (group.content || '').toString().trim();
      if (!content) {
        continue;
      }
      if (groupName) {
        groupBlocks.push(`### ${groupName}\n${content}`);
      } else {
        groupBlocks.push(content);
      }
    }

    const body = groupBlocks.filter(Boolean).join('\n\n');
    if (!body) {
      continue;
    }
    if (sectionName) {
      chunks.push(`## ${sectionName}\n${body}`);
    } else {
      chunks.push(body);
    }
  }

  return chunks.filter(Boolean).join('\n\n');
}

// --- Section: System prompt copied from the original notebook ---
const SYSTEM_PROMPT = `You are a senior Luxury Escapes curation editor. Evaluate each Klook offer for suitability on our platform.
Consider title clarity, image relevance, hero image suitability, category accuracy, description quality, and location correctness.
Return a strict JSON object with keys:
- score (0-5, integer)
- categories (array of categories that best describe the Klook activity. You can only choose from the list below)
- target_audiences (array of target audiences that best describe the Klook activity. You can only choose from the list below)
- hero_image_index (integer index from the numbered image list, starting at 1, or null if no supplied image is suitable)
- hero_image_url (string URL of the hero image that matches the numbered list entry, or null if none are acceptable)
- hero_image_reason (string explaining why the selected image works, or why none are acceptable)
- reason (concise justification including any category recommendations or red flags).

## Categories
These are the possible categories, note each is nested in a parent category. Do not include the parent category in the array.

{
  "Wine & Dine": [
    "Fine dining",
    "Restaurants & bars",
    "Cafés",
    "High tea",
    "Food tours",
    "Wine country trips",
    "Breweries, distilleries & vineyards"
  ],
  "Top Activities": [
    "Yachts, boats & cruises",
    "Cooking classes",
    "Up in the air",
    "Outdoor activities",
    "Watersports",
    "Indoor activities",
    "Photoshoot - Travelshoot",
    "Wildlife Cruises",
    "Cinemas",
    "Golf",
    "Ski",
    "Beach & Pool Clubs",
    "School Holidays"
  ],
  "Attractions & Tickets": [
    "Theme & water parks",
    "Attraction passes",
    "Museums",
    "Zoos & aquariums",
    "Historical sites",
    "Galleries"
  ],
  "Live Events": [
    "Concerts",
    "Theatre",
    "Live sports",
    "Special Events"
  ],
  "Indulge Yourself": [
    "Spa & massage",
    "Hot springs",
    "Wellness"
  ],
  "Lux Exclusives": [
    "The best of the best"
  ],
  "Travel Essentials": [
    "Airport lounges",
    "Luggage",
    "Airport Services",
    "Water Transfers"
  ],
  "Day Tours": [
    "Guided tours",
    "Walking tours",
    "Bike tours",
    "Hop-on-hop-off",
    "Private tours"
  ],
  "Gift Inspiration": [
    "Foodie",
    "Thrill Seeker",
    "Animal Lover",
    "Spa-goer",
    "Family",
    "Aquatic Enthusiast"
  ]
}


## Target Audiences
These are the possible target audiences. Some or all can apply (it is most common for all to apply).
- Solo
- Couple
- Group
- Family

When selecting the hero image:
- You should chose the image most appropriate to be the lead/hero image for the experience offer on our website. This is the image we show in search results, and first on the offer page.
- Use your understanding of the experience based on the offer description, and your knowledge of what customers are looking for, to guide your decision making
- The first image that Klook provided often is very edited to include a promotional overlay. We should not choose that one. (Text naturally in the image, e.g. on the side of a bus, is fine. Edited overlays are not)
- Ideally the customer would be able to look at the image and activity title and think 'I understand what that is about!'
- Use the numbered list of images provided in the prompt; pick the index that best matches the guidance.
- Return hero_image_url as the exact https URL from that list (do not respond with attachment:// references).
- If none of the supplied images are acceptable, set hero_image_index and hero_image_url to null and explain why in hero_image_reason.
`;

function structureActivity(activity: Record<string, JsonValue>, sourcePath: string): StructuredOffer {
  const packagesRaw = Array.isArray(activity.package_list) ? activity.package_list : [];
  const packages: PackageSummary[] = [];
  for (const item of packagesRaw) {
    if (!item || typeof item !== 'object') {
      continue;
    }
    const pkg = item as Record<string, JsonValue>;
    packages.push({
      package_id: pkg.package_id,
      package_name: typeof pkg.package_name === 'string' ? pkg.package_name : undefined,
      sections_markdown: renderSections(pkg.section_info),
    });
  }

  const cityInfo = Array.isArray(activity.city_info) ? activity.city_info : [];
  const primaryCity = (cityInfo[0] || {}) as Record<string, JsonValue>;
  const categoryInfo = (activity.category_info || {}) as Record<string, JsonValue>;
  const status =
    activity.status ||
    activity.curation_status ||
    categoryInfo.curation_status ||
    null;

  const imageDetails = extractImageDetails(activity.images);

  return {
    source_path: sourcePath,
    activity_id: activity.activity_id,
    title: typeof activity.title === 'string' ? activity.title : undefined,
    subtitle: typeof activity.subtitle === 'string' ? activity.subtitle : undefined,
    what_we_love: typeof activity.what_we_love === 'string' ? activity.what_we_love : undefined,
    location: activity.location,
    address: activity.address_desc_multilang,
    category: typeof categoryInfo.sub_category_name === 'string' ? categoryInfo.sub_category_name : undefined,
    category_detail: categoryInfo || null,
    description_markdown: renderSections(activity.section_info),
    packages,
    images: imageDetails.map((detail) => detail.url).filter((url): url is string => Boolean(url)),
    image_details: imageDetails,
    city: typeof primaryCity.city_name === 'string' ? primaryCity.city_name : undefined,
    country: typeof primaryCity.country_name === 'string' ? primaryCity.country_name : undefined,
    status,
    raw: activity,
  };
}

// --- Section: Prompt preparation mirroring the Python helper ---
function summarisePackages(packages: PackageSummary[]): string {
  if (!packages.length) {
    return 'No packages available.';
  }

  const lines: string[] = [];
  packages.forEach((pkg, index) => {
    const name = pkg.package_name || `Package ${index + 1}`;
    const details = pkg.sections_markdown || 'No details supplied.';
    lines.push(`Package: ${name}\n${details}`);
  });
  return lines.join('\n\n');
}

function buildOfferPrompt(offer: StructuredOffer): string {
  const lines: string[] = [
    `Activity ID: ${offer.activity_id ?? 'N/A'}`,
    `Title: ${offer.title || 'N/A'}`,
    `Subtitle: ${offer.subtitle || 'N/A'}`,
    `What we love: ${offer.what_we_love || 'N/A'}`,
    `Location (lat,long): ${offer.location ?? 'N/A'}`,
    `Address: ${offer.address ?? 'N/A'}`,
    `City: ${offer.city || 'N/A'}`,
    `Country: ${offer.country || 'N/A'}`,
    `Current category: ${offer.category || 'N/A'}`,
    '',
    'Offer description markdown:',
    offer.description_markdown || 'No description supplied.',
    '',
    'Packages:',
    summarisePackages(offer.packages),
    '',
    `Images provided (max ${MAX_IMAGES_TO_REVIEW} considered):`,
  ];

  const imageDetails = offer.image_details.slice(0, MAX_IMAGES_TO_REVIEW);
  if (!imageDetails.length) {
    lines.push('No images available.');
  } else {
    lines.push("Image metadata includes Klook's image_type to help avoid stylised banners.");
    lines.push('Always reference these numbers when returning hero_image_index and use the exact URL shown.');
    imageDetails.forEach((image, index) => {
      const altText = (image.alt || '').trim() || 'N/A';
      let description = (image.description || '').trim();
      if (description.length > 120) {
        description = `${description.slice(0, 117)}...`;
      }
      const displayDescription = description || 'N/A';
      const type = image.type || 'UNKNOWN';
      const url = image.url || 'N/A';
      lines.push(`[${index + 1}] type=${type} alt=${altText} desc=${displayDescription} url=${url}`);
    });
  }

  return lines.join('\n');
}

// --- Section: Response parsing helpers for the Responses API ---
function collectResponseText(response: any): string {
  const payload = typeof response?.toJSON === 'function' ? response.toJSON() : response;
  const chunks: string[] = [];
  const outputItems = payload?.output || [];

  for (const item of outputItems) {
    const contents = item?.content || [];
    for (const content of contents) {
      if (content?.type === 'output_text' && typeof content?.text === 'string') {
        chunks.push(content.text);
      }
    }
  }

  return chunks.join('').trim();
}

function parseJsonResponse(text: string): Record<string, JsonValue> {
  if (!text) {
    return { score: null, reason: 'Empty response from model.' } as Record<string, JsonValue>;
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    const match = text.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch {
        // Fall through to error return below.
      }
    }
  }

  return { score: null, reason: `Failed to parse JSON: ${text}` } as Record<string, JsonValue>;
}

function normaliseStringList(value: JsonValue | undefined): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => (item != null ? String(item) : null)).filter((item): item is string => Boolean(item));
  }
  if (value == null || value === '') {
    return [];
  }
  return [String(value)];
}

// --- Section: Core grading routine calling the OpenAI Responses API ---
async function gradeOffer(offer: StructuredOffer): Promise<GradingResult> {
  const client = getClient();
  const prompt = buildOfferPrompt(offer);

  const candidateImages = offer.image_details || [];
  const imagePayload = candidateImages.slice(0, MAX_IMAGES_TO_REVIEW)
    .map((image) => image.url)
    .filter((url): url is string => Boolean(url))
    .map((url) => ({ type: 'input_image', image_url: url }));

  const content = [
    { type: 'input_text', text: prompt },
    ...imagePayload,
  ];

  let responseId: string | null = null;
  let parsed: Record<string, JsonValue> = { score: null, reason: 'No response.' };

  try {
    const response = await client.responses.create({
      model: MODEL_NAME,
      instructions: SYSTEM_PROMPT,
      input: [{ role: 'user', content }],
      reasoning: { effort: REASONING_EFFORT },
      max_output_tokens: MAX_OUTPUT_TOKENS,
      metadata: {
        activity_id: String(offer.activity_id ?? ''),
        activity_title: offer.title || '',
        activity_url: `https://www.klook.com/en-AU/activity/${offer.activity_id ?? ''}`,
        activity_category: offer.category || '',
      },
    });

    responseId = typeof response?.id === 'string' ? response.id : null;
    const responseText = collectResponseText(response);
    parsed = parseJsonResponse(responseText);
  } catch (error: any) {
    const details = error?.response?.data || error?.response?.body || error?.message || error;
    return {
      activity_id: offer.activity_id ?? null,
      score: null,
      reason: `Model call failed: ${JSON.stringify(details)}`,
      categories: [],
      target_audiences: [],
      hero_image_index: null,
      hero_image_url: null,
      hero_image_reason: '',
      response_id: null,
    };
  }

  const categories = normaliseStringList(parsed.categories) ?? [];
  const targetAudiences = normaliseStringList(parsed.target_audiences) ?? [];

  const heroImageIndexRaw = parsed.hero_image_index;
  let heroImageIndex: number | null = null;
  if (typeof heroImageIndexRaw === 'number' && Number.isFinite(heroImageIndexRaw)) {
    heroImageIndex = Math.trunc(heroImageIndexRaw);
  } else if (typeof heroImageIndexRaw === 'string' && heroImageIndexRaw.trim()) {
    const numeric = Number.parseInt(heroImageIndexRaw.trim(), 10);
    heroImageIndex = Number.isFinite(numeric) ? numeric : null;
  }

  let heroImageUrl = parsed.hero_image_url ? String(parsed.hero_image_url).trim() : null;
  if (heroImageUrl === '') {
    heroImageUrl = null;
  }

  if (heroImageIndex != null) {
    if (heroImageIndex >= 1 && heroImageIndex <= candidateImages.length) {
      const candidate = candidateImages[heroImageIndex - 1].url;
      heroImageUrl = candidate || heroImageUrl;
    } else {
      heroImageIndex = null;
    }
  }

  if (heroImageUrl && heroImageUrl.startsWith('attachment://')) {
    heroImageUrl = null;
  }

  if (!heroImageUrl && heroImageIndex != null) {
    const candidate = candidateImages[heroImageIndex - 1]?.url;
    if (candidate) {
      heroImageUrl = candidate;
    }
  }

  if (heroImageUrl && heroImageIndex == null) {
    const idx = candidateImages.findIndex((item) => item.url === heroImageUrl);
    if (idx >= 0) {
      heroImageIndex = idx + 1;
    }
  }

  const heroImageReasonValue = parsed.hero_image_reason;
  const heroImageReason = heroImageReasonValue == null
    ? ''
    : typeof heroImageReasonValue === 'string'
      ? heroImageReasonValue.trim()
      : JSON.stringify(heroImageReasonValue);

  const scoreRaw = parsed.score;
  let score: number | null = null;
  const numericScore = Number(scoreRaw);
  if (!Number.isNaN(numericScore)) {
    score = Math.min(5, Math.max(0, numericScore));
  }

  const reasonValue = parsed.reason != null ? parsed.reason : parsed;
  const reason = typeof reasonValue === 'string' ? reasonValue : JSON.stringify(reasonValue);

  return {
    activity_id: offer.activity_id ?? null,
    score,
    reason,
    categories,
    target_audiences: targetAudiences,
    hero_image_index: heroImageIndex,
    hero_image_url: heroImageUrl,
    hero_image_reason: heroImageReason,
    response_id: responseId,
  };
}

// --- Section: Batch execution with simple concurrency control ---
async function gradeOffers(offers: StructuredOffer[], concurrency: number): Promise<GradingResult[]> {
  if (!offers.length) {
    return [];
  }
  const limiter = Math.max(1, Number.isFinite(concurrency) ? Math.trunc(concurrency) : 1);
  const queue = [...offers];
  const results: GradingResult[] = [];

  async function worker() {
    while (queue.length) {
      const next = queue.shift();
      if (!next) {
        return;
      }
      const result = await gradeOffer(next);
      console.log('Graded offer:', JSON.stringify(result, null, 2));
      results.push(result);
    }
  }

  const workers = Array.from({ length: limiter }, () => worker());
  await Promise.all(workers);

  results.sort((a, b) => {
    const idA = String(a.activity_id ?? '');
    const idB = String(b.activity_id ?? '');
    if (idA === idB) {
      return a.reason.localeCompare(b.reason);
    }
    return idA.localeCompare(idB);
  });

  return results;
}

// --- Section: CSV export matching the Python notebook output columns ---
const CSV_COLUMNS = [
  'activity_id',
  'activity_url',
  'hero_image_index',
  'hero_image_url',
  'hero_image_reason',
  'categories',
  'target_audiences',
  'score',
  'reason',
  'log_url',
];

function escapeCsvValue(value: string | null | number): string {
  const stringValue = value == null ? '' : String(value);
  if (/[",\n]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }
  return stringValue;
}

function buildCsvRow(result: GradingResult): string {
  const activityId = result.activity_id != null ? String(result.activity_id) : '';
  const activityUrl = activityId ? `https://www.klook.com/en-AU/activity/${activityId}` : '';
  const logUrl = result.response_id ? `https://platform.openai.com/logs/${result.response_id}` : '';
  const categories = result.categories.join('; ');
  const audiences = result.target_audiences.join('; ');

  const ordered = [
    activityId,
    activityUrl,
    result.hero_image_index,
    result.hero_image_url,
    result.hero_image_reason,
    categories,
    audiences,
    result.score,
    result.reason,
    logUrl,
  ];

  return ordered.map((item) => escapeCsvValue(item as string | number | null)).join(',');
}

async function writeResultsCsv(results: GradingResult[], outputPath: string): Promise<void> {
  const lines = [CSV_COLUMNS.join(',')];
  for (const result of results) {
    lines.push(buildCsvRow(result));
  }
  await fs.promises.writeFile(outputPath, lines.join('\n'), 'utf-8');
  console.log(`CSV exported to ${outputPath}`);
}

// --- Section: Main execution flow replicating the notebook's orchestration ---
async function main(): Promise<void> {
  console.log(`Loading offers from ${OFFERS_DIR}`);
  const offers = await loadOffers(OFFERS_DIR);
  console.log(`Loaded ${offers.length} offers.`);

  const offersToGrade = offers.filter((offer) => {
    const status = String(offer.status || '').toUpperCase();
    return status !== 'CURATED';
  });
  console.log(`Queued ${offersToGrade.length} offers for grading.`);

  if (!offersToGrade.length) {
    console.log('No offers require grading. Exiting.');
    return;
  }

  const results = await gradeOffers(offersToGrade, CONCURRENCY);
  await writeResultsCsv(results, CSV_OUTPUT_PATH);
}

// Start the script when executed directly.
main().catch((error) => {
  console.error('Fatal error running grading script:', error);
  process.exitCode = 1;
});
