// Pocket TTS ONNX Web Worker
console.log('Pocket TTS Worker Starting...');
self.postMessage({ type: 'status', status: 'Worker Thread Started', state: 'idle' });

// Load ONNX Runtime (will be loaded dynamically in loadModels for module worker)
let ort = null;

// Configuration
const GH_BASE = 'https://media.githubusercontent.com/media/AnEntrypoint/models/main/tts/';
const MODELS = {
    mimi_encoder: GH_BASE + 'mimi_encoder.onnx',
    text_conditioner: GH_BASE + 'text_conditioner.onnx',
    flow_lm_main: GH_BASE + 'flow_lm_main_int8.onnx',
    flow_lm_flow: GH_BASE + 'flow_lm_flow_int8.onnx',
    mimi_decoder: GH_BASE + 'mimi_decoder_int8.onnx',
    tokenizer: GH_BASE + 'tokenizer.model',
    voices: GH_BASE + 'voices.bin'
};

const SAMPLE_RATE = 24000;
const SAMPLES_PER_FRAME = 1920;
const MAX_FRAMES = 500;
const DEBUG_LOGS = true;
// Text chunking target; lower if long passages hit generation limits.
const CHUNK_TARGET_TOKENS = 50;
const CHUNK_GAP_SEC = 0.25;
// If true, re-run voice conditioning per chunk to avoid stale AR state.
const RESET_FLOW_STATE_EACH_CHUNK = true;
// If true, reset decoder state per chunk to avoid carry-over artifacts.
const RESET_MIMI_STATE_EACH_CHUNK = true;

// State
let mimiEncoderSession = null;
let textConditionerSession = null;
let flowLmMainSession = null;
let flowLmFlowSession = null;
let mimiDecoderSession = null;
let tokenizerProcessor = null;
let tokenizerModelB64 = null;
let predefinedVoices = {};
let stTensors = []; // Optimization: Pre-allocated s/t tensors for max LSD
let isGenerating = false;
let isReady = false;

// Dynamic LSD (Latent Solver/Diffusion steps)
const MAX_LSD = 10;  // Default/max quality
let currentLSD = MAX_LSD;

// Current voice embedding (cached)
let currentVoiceEmbedding = null;
let currentVoiceName = null;

// Text preprocessing utilities
const ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const ORDINAL_ONES = ['', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth'];
const ORDINAL_TENS = ['', '', 'twentieth', 'thirtieth', 'fortieth', 'fiftieth', 'sixtieth', 'seventieth', 'eightieth', 'ninetieth'];

function numberToWords(num, options = {}) {
    const { andword = '', zero = 'zero', group = 0 } = options;
    if (num === 0) return zero;
    const convert = (n) => {
        if (n < 20) return ONES[n];
        if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? ' ' + ONES[n % 10] : '');
        if (n < 1000) {
            const remainder = n % 100;
            return ONES[Math.floor(n / 100)] + ' hundred' + (remainder ? (andword ? ' ' + andword + ' ' : ' ') + convert(remainder) : '');
        }
        if (n < 1000000) {
            const thousands = Math.floor(n / 1000);
            const remainder = n % 1000;
            return convert(thousands) + ' thousand' + (remainder ? ' ' + convert(remainder) : '');
        }
        if (n < 1000000000) {
            const millions = Math.floor(n / 1000000);
            const remainder = n % 1000000;
            return convert(millions) + ' million' + (remainder ? ' ' + convert(remainder) : '');
        }
        const billions = Math.floor(n / 1000000000);
        const remainder = n % 1000000000;
        return convert(billions) + ' billion' + (remainder ? ' ' + convert(remainder) : '');
    };
    if (group === 2 && num > 1000 && num < 10000) {
        const high = Math.floor(num / 100);
        const low = num % 100;
        if (low === 0) return convert(high) + ' hundred';
        else if (low < 10) return convert(high) + ' ' + (zero === 'oh' ? 'oh' : zero) + ' ' + ONES[low];
        else return convert(high) + ' ' + convert(low);
    }
    return convert(num);
}

function ordinalToWords(num) {
    if (num < 20) return ORDINAL_ONES[num] || numberToWords(num) + 'th';
    if (num < 100) {
        const tens = Math.floor(num / 10);
        const ones = num % 10;
        if (ones === 0) return ORDINAL_TENS[tens];
        return TENS[tens] + ' ' + ORDINAL_ONES[ones];
    }
    const cardinal = numberToWords(num);
    if (cardinal.endsWith('y')) return cardinal.slice(0, -1) + 'ieth';
    if (cardinal.endsWith('one')) return cardinal.slice(0, -3) + 'first';
    if (cardinal.endsWith('two')) return cardinal.slice(0, -3) + 'second';
    if (cardinal.endsWith('three')) return cardinal.slice(0, -5) + 'third';
    if (cardinal.endsWith('ve')) return cardinal.slice(0, -2) + 'fth';
    if (cardinal.endsWith('e')) return cardinal.slice(0, -1) + 'th';
    if (cardinal.endsWith('t')) return cardinal + 'h';
    return cardinal + 'th';
}

const UNICODE_MAP = {
    'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a', 'æ': 'ae', 'ç': 'c', 'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e', 'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i', 'ñ': 'n', 'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o', 'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ý': 'y', 'ÿ': 'y', 'ß': 'ss', 'œ': 'oe', 'ð': 'd', 'þ': 'th', 'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A', 'Å': 'A', 'Æ': 'AE', 'Ç': 'C', 'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E', 'Ì': 'I', 'Í': 'I', 'Î': 'I', 'Ï': 'I', 'Ñ': 'N', 'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O', 'Ø': 'O', 'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U', 'Ý': 'Y', '\u201C': '"', '\u201D': '"', '\u2018': "'", '\u2019': "'", '\u2026': '...', '\u2013': '-', '\u2014': '-'
};

function convertToAscii(text) {
    return text.split('').map(c => UNICODE_MAP[c] || c).join('').normalize('NFD').replace(/[\u0300-\u036f]/g, '');
}

const ABBREVIATIONS = [
    [/\bmrs\./gi, 'misuss'], [/\bms\./gi, 'miss'], [/\bmr\./gi, 'mister'], [/\bdr\./gi, 'doctor'], [/\bst\./gi, 'saint'], [/\bco\./gi, 'company'], [/\bjr\./gi, 'junior'], [/\bmaj\./gi, 'major'], [/\bgen\./gi, 'general'], [/\bdrs\./gi, 'doctors'], [/\brev\./gi, 'reverend'], [/\blt\./gi, 'lieutenant'], [/\bhon\./gi, 'honorable'], [/\bsgt\./gi, 'sergeant'], [/\bcapt\./gi, 'captain'], [/\besq\./gi, 'esquire'], [/\bltd\./gi, 'limited'], [/\bcol\./gi, 'colonel'], [/\bft\./gi, 'fort']
];
const CASED_ABBREVIATIONS = [
    [/\bTTS\b/g, 'text to speech'], [/\bHz\b/g, 'hertz'], [/\bkHz\b/g, 'kilohertz'], [/\bKBs\b/g, 'kilobytes'], [/\bKB\b/g, 'kilobyte'], [/\bMBs\b/g, 'megabytes'], [/\bMB\b/g, 'megabyte'], [/\bGBs\b/g, 'gigabytes'], [/\bGB\b/g, 'gigabyte'], [/\bTBs\b/g, 'terabytes'], [/\bTB\b/g, 'terabyte'], [/\bAPIs\b/g, "a p i's"], [/\bAPI\b/g, 'a p i'], [/\bCLIs\b/g, "c l i's"], [/\bCLI\b/g, 'c l i'], [/\bCPUs\b/g, "c p u's"], [/\bCPU\b/g, 'c p u'], [/\bGPUs\b/g, "g p u's"], [/\bGPU\b/g, 'g p u'], [/\bAve\b/g, 'avenue'], [/\betc\b/g, 'etcetera']
];

function expandAbbreviations(text) {
    for (const [regex, replacement] of [...ABBREVIATIONS, ...CASED_ABBREVIATIONS]) text = text.replace(regex, replacement);
    return text;
}

const NUM_PREFIX_RE = /#(\d)/g;
const NUM_SUFFIX_RE = /(\d)([KMBT])/gi;
const NUM_LETTER_SPLIT_RE = /(\d)([a-z])|([a-z])(\d)/gi;
const COMMA_NUMBER_RE = /(\d[\d,]+\d)/g;
const DATE_RE = /(^|[^/])(\d\d?[/-]\d\d?[/-]\d\d(?:\d\d)?)($|[^/])/g;
const PHONE_NUMBER_RE = /\(?\d{3}\)?[-.\s]\d{3}[-.\s]?\d{4}/g;
const TIME_RE = /(\d\d?):(\d\d)(?::(\d\d))?/g;
const POUNDS_RE = /£([\d,]*\d+)/g;
const DOLLARS_RE = /\$([\d.,]*\d+)/g;
const DECIMAL_NUMBER_RE = /(\d+(?:\.\d+)+)/g;
const MULTIPLY_RE = /(\d)\s?\*\s?(\d)/g;
const DIVIDE_RE = /(\d)\s?\/\s?(\d)/g;
const ADD_RE = /(\d)\s?\+\s?(\d)/g;
const SUBTRACT_RE = /(\d)?\s?-\s?(\d)/g;
const FRACTION_RE = /(\d+)\/(\d+)/g;
const ORDINAL_RE = /(\d+)(st|nd|rd|th)/gi;
const NUMBER_RE = /\d+/g;

function normalizeNumbers(text) {
    text = text.replace(NUM_PREFIX_RE, (_, d) => `number ${d}`);
    text = text.replace(NUM_SUFFIX_RE, (_, num, suffix) => {
        const map = { k: 'thousand', m: 'million', b: 'billion', t: 'trillion' };
        return `${num} ${map[suffix.toLowerCase()]}`;
    });
    for (let i = 0; i < 2; i++) {
        text = text.replace(NUM_LETTER_SPLIT_RE, (m, d1, l1, l2, d2) => {
            if (d1 && l1) return `${d1} ${l1}`;
            if (l2 && d2) return `${l2} ${d2}`;
            return m;
        });
    }
    text = text.replace(COMMA_NUMBER_RE, m => m.replace(/,/g, ''));
    text = text.replace(DATE_RE, (_, pre, date, post) => pre + date.split(/[./-]/).join(' dash ') + post);
    text = text.replace(PHONE_NUMBER_RE, m => {
        const digits = m.replace(/\D/g, '');
        return digits.length === 10 ? `${digits.slice(0, 3).split('').join(' ')}, ${digits.slice(3, 6).split('').join(' ')}, ${digits.slice(6).split('').join(' ')}` : m;
    });
    text = text.replace(TIME_RE, (_, hours, minutes, seconds) => {
        const h = parseInt(hours), m = parseInt(minutes), s = seconds ? parseInt(seconds) : 0;
        if (!seconds) return m === 0 ? (h === 0 ? '0' : h > 12 ? `${hours} minutes` : `${hours} o'clock`) : minutes.startsWith('0') ? `${hours} oh ${minutes[1]}` : `${hours} ${minutes}`;
        let res = '';
        if (h !== 0) res = hours + ' ' + (m === 0 ? 'oh oh' : minutes.startsWith('0') ? `oh ${minutes[1]}` : minutes);
        else if (m !== 0) res = minutes + ' ' + (s === 0 ? 'oh oh' : seconds.startsWith('0') ? `oh ${seconds[1]}` : seconds);
        else res = seconds;
        return res + ' ' + (s === 0 ? '' : seconds.startsWith('0') ? `oh ${seconds[1]}` : seconds);
    });
    text = text.replace(POUNDS_RE, (_, amount) => `${amount.replace(/,/g, '')} pounds`);
    text = text.replace(DOLLARS_RE, (_, amount) => {
        const parts = amount.replace(/,/g, '').split('.');
        const dollars = parseInt(parts[0]) || 0;
        const cents = parts[1] ? parseInt(parts[1]) : 0;
        if (dollars && cents) return `${dollars} ${dollars === 1 ? 'dollar' : 'dollars'}, ${cents} ${cents === 1 ? 'cent' : 'cents'}`;
        if (dollars) return `${dollars} ${dollars === 1 ? 'dollar' : 'dollars'}`;
        if (cents) return `${cents} ${cents === 1 ? 'cent' : 'cents'}`;
        return 'zero dollars';
    });
    text = text.replace(DECIMAL_NUMBER_RE, m => m.split('.').join(' point ').split('').join(' '));
    text = text.replace(MULTIPLY_RE, '$1 times $2');
    text = text.replace(DIVIDE_RE, '$1 over $2');
    text = text.replace(ADD_RE, '$1 plus $2');
    text = text.replace(SUBTRACT_RE, (_, a, b) => (a ? a : '') + ' minus ' + b);
    text = text.replace(FRACTION_RE, '$1 over $2');
    text = text.replace(ORDINAL_RE, (_, num) => ordinalToWords(parseInt(num)));
    text = text.replace(NUMBER_RE, m => {
        const num = parseInt(m);
        if (num > 1000 && num < 3000) {
            if (num === 2000) return 'two thousand';
            if (num > 2000 && num < 2010) return 'two thousand ' + numberToWords(num % 100);
            if (num % 100 === 0) return numberToWords(Math.floor(num / 100)) + ' hundred';
            return numberToWords(num, { zero: 'oh', group: 2 });
        }
        return numberToWords(num);
    });
    return text;
}

const SPECIAL_CHARACTERS = [
    [/@/g, ' at '], [/&/g, ' and '], [/%/g, ' percent '], [/:/g, '.'], [/;/g, ','], [/\+/g, ' plus '], [/\\/g, ' backslash '], [/~/g, ' about '], [/(^| )<3/g, ' heart '], [/<=/g, ' less than or equal to '], [/>=/g, ' greater than or equal to '], [/</g, ' less than '], [/>/g, ' greater than '], [/=/g, ' equals '], [/\//g, ' slash '], [/_/g, ' '],
];
const LINK_HEADER_RE = /https?:\/\//gi;
const DASH_RE = /(.) - (.)/g;
const DOT_RE = /([A-Z])\.([A-Z])/gi;
const PARENTHESES_RE = /[\(\[\{][^\)\]\}]*[\)\]\}](.)?/g;

function normalizeSpecial(text) {
    text = text.replace(LINK_HEADER_RE, 'h t t p s colon slash slash ');
    text = text.replace(DASH_RE, '$1, $2');
    text = text.replace(DOT_RE, '$1 dot $2');
    text = text.replace(PARENTHESES_RE, (m, after) => {
        let result = m.replace(/[\(\[\{]/g, ', ').replace(/[\)\]\}]/g, ', ');
        if (after && /[$.!?,]/.test(after)) result = result.slice(0, -2) + after;
        return result;
    });
    return text;
}

function expandSpecialCharacters(text) {
    for (const [regex, replacement] of SPECIAL_CHARACTERS) text = text.replace(regex, replacement);
    return text;
}

function collapseWhitespace(text) {
    return text.replace(/\s+/g, ' ').replace(/ ([.\?!,])/g, '$1');
}

function dedupPunctuation(text) {
    return text.replace(/\.\.\.+/g, '[ELLIPSIS]').replace(/,+/g, ',').replace(/[.,]*\.[.,]*/g, '.').replace(/[.,!]*![.,!]*/g, '!').replace(/[.,!?]*\?[.,!?]*/g, '?').replace(/\[ELLIPSIS\]/g, '...');
}

const SENTENCE_SPLIT_RE = /[^.!?]+[.!?]+|[^.!?]+$/g;

function splitTextIntoSentences(text) {
    const matches = text.match(SENTENCE_SPLIT_RE);
    if (!matches) return [];
    return matches.map(sentence => sentence.trim()).filter(Boolean);
}

function splitTokenIdsIntoChunks(tokenIds, maxTokens) {
    const chunks = [];
    for (let i = 0; i < tokenIds.length; i += maxTokens) {
        const chunkText = tokenizerProcessor.decodeIds(tokenIds.slice(i, i + maxTokens)).trim();
        if (chunkText) chunks.push(chunkText);
    }
    return chunks;
}

// Split text into sentence chunks (target <= CHUNK_TARGET_TOKENS tokens)
function splitIntoBestSentences(text) {
    const preparedText = prepareText(text);
    if (!preparedText) return [];

    const sentences = splitTextIntoSentences(preparedText);
    if (sentences.length === 0) return [];

    // Merge sentences into chunks that stay within the token target
    const chunks = [];
    let currentChunk = '';
    for (const sentenceText of sentences) {
        const sentenceTokenIds = tokenizerProcessor.encodeIds(sentenceText);
        const sentenceTokens = sentenceTokenIds.length;

        if (sentenceTokens > CHUNK_TARGET_TOKENS) {
            if (currentChunk !== '') {
                chunks.push(currentChunk.trim());
                currentChunk = '';
            }
            const splitChunks = splitTokenIdsIntoChunks(sentenceTokenIds, CHUNK_TARGET_TOKENS);
            for (const splitChunk of splitChunks) {
                if (splitChunk) chunks.push(splitChunk.trim());
            }
            continue;
        }

        if (currentChunk === '') {
            currentChunk = sentenceText;
            continue;
        }

        const combined = `${currentChunk} ${sentenceText}`;
        const combinedTokens = tokenizerProcessor.encodeIds(combined).length;
        if (combinedTokens > CHUNK_TARGET_TOKENS) {
            chunks.push(currentChunk.trim());
            currentChunk = sentenceText;
        } else {
            currentChunk = combined;
        }
    }

    if (currentChunk !== '') {
        chunks.push(currentChunk.trim());
    }

    return chunks;
}

// Pocket TTS specific text preprocessing
function prepareText(text) {
    text = text.trim();
    if (!text) return '';

    // Convert to ASCII
    text = convertToAscii(text);

    // Normalize numbers first
    text = normalizeNumbers(text);

    // Normalize special characters
    text = normalizeSpecial(text);

    // Expand abbreviations
    text = expandAbbreviations(text);

    // Expand special characters
    text = expandSpecialCharacters(text);

    // Collapse whitespace
    text = collapseWhitespace(text);

    // Deduplicate punctuation
    text = dedupPunctuation(text);

    // Final cleanup
    text = text.trim();

    // Ensure proper punctuation at end
    if (text && text[text.length - 1].match(/[a-zA-Z0-9]/)) {
        text = text + '.';
    }

    // Capitalize first letter
    if (text && !text[0].match(/[A-Z]/)) {
        text = text[0].toUpperCase() + text.slice(1);
    }

    return text;
}

// ----------------------------------------------------------------------------
// Worker Logic
// ----------------------------------------------------------------------------

self.onmessage = async (e) => {
    const { type, data } = e.data;
    console.log('Worker received message:', type);

    if (type === 'load') {
        try {
            await loadModels();
        } catch (err) {
            postMessage({ type: 'error', error: err.toString() });
        }
    } else if (type === 'generate') {
        if (!isReady) {
            postMessage({ type: 'error', error: 'Models are not loaded yet.' });
            return;
        }
        if (isGenerating) return;
        try {
            await startGeneration(data.text, data.voice);
        } catch (err) {
            console.error('Generation Error:', err);
            postMessage({ type: 'error', error: err.toString() });
        }
    } else if (type === 'encode_voice') {
        if (!isReady) {
            postMessage({ type: 'error', error: 'Models are not loaded yet.' });
            return;
        }
        try {
            const embedding = await encodeVoiceAudio(data.audio);
            currentVoiceEmbedding = embedding;
            currentVoiceName = 'custom';
            postMessage({ type: 'voice_encoded', voiceName: 'custom' });
        } catch (err) {
            console.error('Voice encoding error:', err);
            postMessage({ type: 'error', error: 'Failed to encode voice: ' + err.toString() });
        }
    } else if (type === 'set_voice') {
        if (!isReady) {
            postMessage({ type: 'error', error: 'Models are not loaded yet.' });
            return;
        }
        if (data.voiceName === 'custom') {
            // Custom voice already set via encode_voice
            postMessage({ type: 'voice_set', voiceName: 'custom' });
        } else if (predefinedVoices[data.voiceName]) {
            currentVoiceEmbedding = predefinedVoices[data.voiceName];
            currentVoiceName = data.voiceName;
            postMessage({ type: 'voice_set', voiceName: data.voiceName });
        } else {
            postMessage({ type: 'error', error: `Unknown voice: ${data.voiceName}` });
        }
    } else if (type === 'set_lsd') {
        // Dynamic LSD adjustment for edge devices
        const newLSD = Math.max(1, Math.min(MAX_LSD, data.lsd));
        if (newLSD !== currentLSD) {
            console.log(`LSD adjusted: ${currentLSD} → ${newLSD}`);
            currentLSD = newLSD;
        }
    } else if (type === 'stop') {
        isGenerating = false;
        postMessage({ type: 'status', status: 'Stopped', state: 'idle' });
    }
};


function decodeWavTo24k(buffer) {
    const view = new DataView(buffer);
    const channels = view.getUint16(22, true);
    const sampleRate = view.getUint32(24, true);
    const numSamples = Math.floor((buffer.byteLength - 44) / 2 / channels);
    const mono = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
        mono[i] = view.getInt16(44 + i * channels * 2, true) / 32768;
    }
    if (sampleRate === 24000) return mono;
    const ratio = sampleRate / 24000;
    const out = new Float32Array(Math.round(numSamples / ratio));
    for (let i = 0; i < out.length; i++) {
        const idx = i * ratio;
        const lo = Math.floor(idx);
        const hi = Math.min(lo + 1, numSamples - 1);
        out[i] = mono[lo] * (1 - (idx - lo)) + mono[hi] * (idx - lo);
    }
    return out;
}

async function loadModels() {
    if (mimiEncoderSession) return;

    postMessage({ type: 'status', status: 'Loading ONNX Runtime...', state: 'loading' });

    // Load ONNX Runtime dynamically
    const version = '1.20.0';
    const cdnBase = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/`;

    try {
        const ortModule = await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/ort.min.mjs`);
        ort = ortModule.default || ortModule;
    } catch (e) {
        console.error('Failed to load ONNX Runtime:', e);
        throw new Error('Failed to load ONNX Runtime: ' + e.message);
    }

    if (!ort) {
        throw new Error('ONNX Runtime failed to load');
    }

    postMessage({ type: 'status', status: 'Loading models...', state: 'loading' });

    // Configure WASM Paths
    ort.env.wasm.wasmPaths = cdnBase;

    // Enable SIMD for significant performance boost (2-4x faster)
    ort.env.wasm.simd = true;

    // Configure multi-threading
    if (!self.crossOriginIsolated) {
        console.warn('Environment is not cross-origin isolated. Disabling WASM multi-threading.');
        console.warn('To enable multi-threading, serve with headers:');
        console.warn('  Cross-Origin-Opener-Policy: same-origin');
        console.warn('  Cross-Origin-Embedder-Policy: require-corp');
        ort.env.wasm.numThreads = 1;
    } else {
        const threads = Math.min(navigator.hardwareConcurrency || 4, 8);
        ort.env.wasm.numThreads = threads;
        if (DEBUG_LOGS) {
            console.log(`Multi-threading enabled with ${threads} threads`);
        }
    }

    console.log(`ORT: crossOriginIsolated=${self.crossOriginIsolated}, simd=${ort.env.wasm.simd}, threads=${ort.env.wasm.numThreads}`);

    try {
        const sessionOptions = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };

        // Load all models in parallel
        postMessage({ type: 'status', status: 'Loading MIMI encoder...', state: 'loading' });
        if (DEBUG_LOGS) {
            console.log('Loading MIMI encoder...');
        }

        const [encoderRes, textCondRes, flowMainRes, flowFlowRes, decoderRes] = await Promise.all([
            ort.InferenceSession.create(MODELS.mimi_encoder, sessionOptions),
            ort.InferenceSession.create(MODELS.text_conditioner, sessionOptions),
            ort.InferenceSession.create(MODELS.flow_lm_main, sessionOptions),
            ort.InferenceSession.create(MODELS.flow_lm_flow, sessionOptions),
            ort.InferenceSession.create(MODELS.mimi_decoder, sessionOptions)
        ]);

        mimiEncoderSession = encoderRes;
        textConditionerSession = textCondRes;
        flowLmMainSession = flowMainRes;
        flowLmFlowSession = flowFlowRes;
        mimiDecoderSession = decoderRes;

        if (DEBUG_LOGS) {
            console.log('All models loaded successfully');
            console.log('Flow LM Main inputs:', flowLmMainSession.inputNames);
            console.log('Flow LM Main outputs:', flowLmMainSession.outputNames);
            console.log('MIMI decoder inputs:', mimiDecoderSession.inputNames);
            console.log('MIMI decoder outputs:', mimiDecoderSession.outputNames);
        }

        // Load tokenizer
        postMessage({ type: 'status', status: 'Loading tokenizer...', state: 'loading' });
        if (DEBUG_LOGS) {
            console.log('Loading tokenizer...');
        }

        const tokenizerResponse = await fetch(MODELS.tokenizer);
        if (!tokenizerResponse.ok) {
            throw new Error(`Failed to load tokenizer: ${tokenizerResponse.statusText}`);
        }
        const tokenizerBuffer = await tokenizerResponse.arrayBuffer();
        const tokBytes = new Uint8Array(tokenizerBuffer);
        let tokBin = '';
        for (let i = 0; i < tokBytes.length; i += 8192) {
            tokBin += String.fromCharCode(...tokBytes.subarray(i, i + 8192));
        }
        tokenizerModelB64 = btoa(tokBin);

        // Import and initialize sentencepiece processor
        const spModule = await import('https://esm.sh/@sctg/sentencepiece-js@1.3.3');
        const SentencePieceProcessor = spModule.SentencePieceProcessor;
        if (!SentencePieceProcessor) {
            throw new Error('SentencePieceProcessor not found in sentencepiece.js');
        }
        tokenizerProcessor = new SentencePieceProcessor();
        await tokenizerProcessor.loadFromB64StringModel(tokenizerModelB64);
        if (DEBUG_LOGS) {
            console.log('Tokenizer loaded');
        }

        // Load predefined voices
        postMessage({ type: 'status', status: 'Loading voices...', state: 'loading' });
        if (DEBUG_LOGS) {
            console.log('Loading predefined voices...');
        }

        try {
            const voicesResponse = await fetch(MODELS.voices);
            if (voicesResponse.ok) {
                const voicesData = await voicesResponse.arrayBuffer();
                predefinedVoices = parseVoicesBin(voicesData);
                if (DEBUG_LOGS) {
                    console.log('Loaded voices:', Object.keys(predefinedVoices));
                }

                // Set default voice
                if (predefinedVoices['cosette']) {
                    currentVoiceEmbedding = predefinedVoices['cosette'];
                    currentVoiceName = 'cosette';
                } else {
                    // Use first available voice
                    const firstVoice = Object.keys(predefinedVoices)[0];
                    if (firstVoice) {
                        currentVoiceEmbedding = predefinedVoices[firstVoice];
                        currentVoiceName = firstVoice;
                    }
                }
            } else {
                console.warn(`[TTS] voices.bin not found (HTTP ${voicesResponse.status}), will use cleetus.wav fallback`);
            }
        } catch (e) {
            console.error('[TTS] Could not load predefined voices:', e);
            postMessage({ type: 'status', status: 'Voice load error: ' + e.message, state: 'loading' });
        }

        // Send list of available voices
        postMessage({
            type: 'voices_loaded',
            voices: Object.keys(predefinedVoices),
            defaultVoice: currentVoiceName
        });

        if (!currentVoiceEmbedding) {
            console.log('[TTS] No predefined voice available, encoding cleetus.wav fallback...');
            try {
                const wavRes = await fetch('./cleetus.wav');
                if (wavRes.ok) {
                    const wavBuf = await wavRes.arrayBuffer();
                    const audio = decodeWavTo24k(wavBuf);
                    console.log(`[TTS] Encoding cleetus.wav (${audio.length} samples at 24kHz)...`);
                    currentVoiceEmbedding = await encodeVoiceAudio(audio);
                    currentVoiceName = 'cleetus';
                    console.log('[TTS] cleetus.wav voice encoded successfully, shape:', currentVoiceEmbedding.shape);
                    postMessage({ type: 'voices_loaded', voices: ['cleetus'], defaultVoice: 'cleetus' });
                } else {
                    console.error(`[TTS] cleetus.wav not found (HTTP ${wavRes.status})`);
                }
            } catch(e) {
                console.error('[TTS] Failed to encode default voice:', e.message, e.stack);
            }
        }

        if (!currentVoiceEmbedding) {
            console.warn('[TTS] No voice embedding loaded — worker will be ready but voice must be uploaded before generating');
            postMessage({ type: 'status', status: 'No voice loaded — upload audio to set voice', state: 'loading' });
        }

        // Pre-allocate s/t tensors for Flow Matching Loop (Optimization)
        // Pre-allocate for MAX_LSD to support dynamic switching
        if (DEBUG_LOGS) {
            console.log(`Pre-allocating Flow Matching tensors for LSD 1-${MAX_LSD}...`);
        }
        stTensors = {};

        for (let lsd = 1; lsd <= MAX_LSD; lsd++) {
            stTensors[lsd] = [];
            const dt = 1.0 / lsd;
            for (let j = 0; j < lsd; j++) {
                const s = j / lsd;
                const t = s + dt;
                stTensors[lsd].push({
                    s: new ort.Tensor('float32', new Float32Array([s]), [1, 1]),
                    t: new ort.Tensor('float32', new Float32Array([t]), [1, 1])
                });
            }
        }

        isReady = true;
        postMessage({ type: 'status', status: 'Ready', state: 'idle' });
        postMessage({ type: 'model_status', status: 'ready', text: 'Ready' });
        postMessage({ type: 'loaded' });

    } catch (err) {
        console.error('Model load failed:', err);
        throw err;
    }
}

function parseVoicesBin(buffer) {
    // Simple binary format:
    // Header: 4 bytes (uint32) = number of voices
    // For each voice:
    //   - 32 bytes: voice name (null-terminated string)
    //   - 4 bytes (uint32): number of frames
    //   - 4 bytes (uint32): embedding dim (1024)
    //   - frames * dim * 4 bytes: float32 embeddings

    const voices = {};
    const view = new DataView(buffer);
    let offset = 0;

    const numVoices = view.getUint32(offset, true);
    offset += 4;

    for (let i = 0; i < numVoices; i++) {
        // Read voice name
        const nameBytes = new Uint8Array(buffer, offset, 32);
        const nameEnd = nameBytes.indexOf(0);
        const name = new TextDecoder().decode(nameBytes.subarray(0, nameEnd > 0 ? nameEnd : 32)).trim();
        offset += 32;

        // Read dimensions
        const numFrames = view.getUint32(offset, true);
        offset += 4;
        const embDim = view.getUint32(offset, true);
        offset += 4;

        // Read embeddings (copy to avoid alignment issues)
        const embSize = numFrames * embDim;
        const embBytes = new Uint8Array(buffer, offset, embSize * 4);
        const embeddings = new Float32Array(embBytes.buffer.slice(offset, offset + embSize * 4));
        offset += embSize * 4;

        voices[name] = {
            data: embeddings,
            shape: [1, numFrames, embDim]
        };

        console.log(`Loaded voice '${name}': ${numFrames} frames, ${embDim} dim`);
    }

    return voices;
}

async function encodeVoiceAudio(audioData) {
    // audioData should be Float32Array at 24kHz, mono
    // Reshape to [1, 1, samples]
    const input = new ort.Tensor('float32', audioData, [1, 1, audioData.length]);

    const outputs = await mimiEncoderSession.run({ audio: input });
    const embeddings = outputs[mimiEncoderSession.outputNames[0]];

    return {
        data: new Float32Array(embeddings.data),
        shape: embeddings.dims
    };
}

// Hardcoded state shapes extracted from ONNX model metadata
// These are the initial shapes - dynamic dimensions start at 0
const FLOW_LM_STATE_SHAPES = {
    // KV cache layers: [kv=2, batch=1, max_seq=1000, heads=16, head_dim=64]
    state_0: { shape: [2, 1, 1000, 16, 64], dtype: 'float32' },
    state_1: { shape: [0], dtype: 'float32' },  // dynamic
    state_2: { shape: [1], dtype: 'int64' },    // step counter
    state_3: { shape: [2, 1, 1000, 16, 64], dtype: 'float32' },
    state_4: { shape: [0], dtype: 'float32' },
    state_5: { shape: [1], dtype: 'int64' },
    state_6: { shape: [2, 1, 1000, 16, 64], dtype: 'float32' },
    state_7: { shape: [0], dtype: 'float32' },
    state_8: { shape: [1], dtype: 'int64' },
    state_9: { shape: [2, 1, 1000, 16, 64], dtype: 'float32' },
    state_10: { shape: [0], dtype: 'float32' },
    state_11: { shape: [1], dtype: 'int64' },
    state_12: { shape: [2, 1, 1000, 16, 64], dtype: 'float32' },
    state_13: { shape: [0], dtype: 'float32' },
    state_14: { shape: [1], dtype: 'int64' },
    state_15: { shape: [2, 1, 1000, 16, 64], dtype: 'float32' },
    state_16: { shape: [0], dtype: 'float32' },
    state_17: { shape: [1], dtype: 'int64' },
};

const MIMI_DECODER_STATE_SHAPES = {
    state_0: { shape: [1], dtype: 'bool' },
    state_1: { shape: [1, 512, 6], dtype: 'float32' },
    state_2: { shape: [1], dtype: 'bool' },
    state_3: { shape: [1, 64, 2], dtype: 'float32' },
    state_4: { shape: [1, 256, 6], dtype: 'float32' },
    state_5: { shape: [1], dtype: 'bool' },
    state_6: { shape: [1, 256, 2], dtype: 'float32' },
    state_7: { shape: [1], dtype: 'bool' },
    state_8: { shape: [1, 128, 0], dtype: 'float32' },  // dynamic
    state_9: { shape: [1, 128, 5], dtype: 'float32' },
    state_10: { shape: [1], dtype: 'bool' },
    state_11: { shape: [1, 128, 2], dtype: 'float32' },
    state_12: { shape: [1], dtype: 'bool' },
    state_13: { shape: [1, 64, 0], dtype: 'float32' },  // dynamic
    state_14: { shape: [1, 64, 4], dtype: 'float32' },
    state_15: { shape: [1], dtype: 'bool' },
    state_16: { shape: [1, 64, 2], dtype: 'float32' },
    state_17: { shape: [1], dtype: 'bool' },
    state_18: { shape: [1, 32, 0], dtype: 'float32' },  // dynamic
    state_19: { shape: [2, 1, 8, 1000, 64], dtype: 'float32' },
    state_20: { shape: [1], dtype: 'int64' },
    state_21: { shape: [1], dtype: 'int64' },
    state_22: { shape: [2, 1, 8, 1000, 64], dtype: 'float32' },
    state_23: { shape: [1], dtype: 'int64' },
    state_24: { shape: [1], dtype: 'int64' },
    state_25: { shape: [1], dtype: 'bool' },
    state_26: { shape: [1, 512, 16], dtype: 'float32' },
    state_27: { shape: [1], dtype: 'bool' },
    state_28: { shape: [1, 1, 6], dtype: 'float32' },
    state_29: { shape: [1], dtype: 'bool' },
    state_30: { shape: [1, 64, 2], dtype: 'float32' },
    state_31: { shape: [1], dtype: 'bool' },
    state_32: { shape: [1, 32, 0], dtype: 'float32' },  // dynamic
    state_33: { shape: [1], dtype: 'bool' },
    state_34: { shape: [1, 512, 2], dtype: 'float32' },
    state_35: { shape: [1], dtype: 'bool' },
    state_36: { shape: [1, 64, 4], dtype: 'float32' },
    state_37: { shape: [1], dtype: 'bool' },
    state_38: { shape: [1, 128, 2], dtype: 'float32' },
    state_39: { shape: [1], dtype: 'bool' },
    state_40: { shape: [1, 64, 0], dtype: 'float32' },  // dynamic
    state_41: { shape: [1], dtype: 'bool' },
    state_42: { shape: [1, 128, 5], dtype: 'float32' },
    state_43: { shape: [1], dtype: 'bool' },
    state_44: { shape: [1, 256, 2], dtype: 'float32' },
    state_45: { shape: [1], dtype: 'bool' },
    state_46: { shape: [1, 128, 0], dtype: 'float32' },  // dynamic
    state_47: { shape: [1], dtype: 'bool' },
    state_48: { shape: [1, 256, 6], dtype: 'float32' },
    state_49: { shape: [2, 1, 8, 1000, 64], dtype: 'float32' },
    state_50: { shape: [1], dtype: 'int64' },
    state_51: { shape: [1], dtype: 'int64' },
    state_52: { shape: [2, 1, 8, 1000, 64], dtype: 'float32' },
    state_53: { shape: [1], dtype: 'int64' },
    state_54: { shape: [1], dtype: 'int64' },
    state_55: { shape: [1, 512, 16], dtype: 'float32' },
};

function initState(session, stateShapes) {
    /**
     * Initialize state tensors for a stateful ONNX model using hardcoded shapes.
     */
    const state = {};

    for (const inputName of session.inputNames) {
        if (inputName.startsWith('state_')) {
            const stateInfo = stateShapes[inputName];
            if (!stateInfo) {
                console.warn(`Unknown state input: ${inputName}, skipping`);
                continue;
            }

            const { shape, dtype } = stateInfo;
            const size = shape.reduce((a, b) => a * b, 1);

            let data;
            if (dtype === 'int64') {
                data = new BigInt64Array(size);
            } else if (dtype === 'bool') {
                data = new Uint8Array(size);
            } else {
                data = new Float32Array(size);
            }

            state[inputName] = new ort.Tensor(dtype, data, shape);
            if (DEBUG_LOGS) {
                console.log(`Init state ${inputName}: shape=${JSON.stringify(shape)}, dtype=${dtype}`);
            }
        }
    }

    return state;
}

async function startGeneration(text, voiceName) {
    isGenerating = true;
    currentLSD = MAX_LSD;  // Reset to max quality for each new generation
    postMessage({ type: 'status', status: 'Generating...', state: 'running' });
    postMessage({ type: 'generation_started', data: { time: performance.now() } });

    try {
        // Split text into sentence chunks (target <= CHUNK_TARGET_TOKENS tokens)
        const chunks = splitIntoBestSentences(text);
        console.log(`Split into ${chunks.length} chunks:`, chunks);

        if (chunks.length === 0) {
            throw new Error('No text to generate');
        }

        // Get voice embedding
        let voiceEmb = currentVoiceEmbedding;
        if (voiceName && voiceName !== currentVoiceName) {
            if (predefinedVoices[voiceName]) {
                voiceEmb = predefinedVoices[voiceName];
                currentVoiceEmbedding = voiceEmb;
                currentVoiceName = voiceName;
            }
        }

        if (!voiceEmb) {
            throw new Error('No voice embedding available. Please select a voice or upload custom audio.');
        }

        // Run generation pipeline with chunks
        await runGenerationPipeline(voiceEmb, chunks);

    } catch (err) {
        console.error('Generation error:', err);
        postMessage({ type: 'error', error: err.toString() });
    } finally {
        if (isGenerating) {
            postMessage({ type: 'stream_ended' });
            postMessage({ type: 'status', status: 'Finished', state: 'idle' });
        }
        isGenerating = false;
    }
}

async function runGenerationPipeline(voiceEmb, chunks) {
    // Initialize state - may be reset per chunk
    let mimiState = initState(mimiDecoderSession, MIMI_DECODER_STATE_SHAPES);
    const emptySeq = new ort.Tensor('float32', new Float32Array(0), [1, 0, 32]);
    const emptyTextEmb = new ort.Tensor('float32', new Float32Array(0), [1, 0, 1024]);

    // Voice embedding tensor
    const voiceTensor = new ort.Tensor('float32', voiceEmb.data, voiceEmb.shape);
    console.log('Voice embeddings shape:', voiceEmb.shape);

    async function buildVoiceConditionedState() {
        let flowLmState = initState(flowLmMainSession, FLOW_LM_STATE_SHAPES);
        console.log('Running voice conditioning...');
        const voiceCondInputs = {
            sequence: emptySeq,
            text_embeddings: voiceTensor,
            ...flowLmState
        };

        let condResult = await flowLmMainSession.run(voiceCondInputs);

        // Update state from voice conditioning
        for (let i = 2; i < flowLmMainSession.outputNames.length; i++) {
            const outputName = flowLmMainSession.outputNames[i];
            if (outputName.startsWith('out_state_')) {
                const stateIdx = parseInt(outputName.replace('out_state_', ''));
                flowLmState[`state_${stateIdx}`] = condResult[outputName];
            }
        }
        return flowLmState;
    }

    let flowLmState = await buildVoiceConditionedState();

    // Streaming parameters
    const FIRST_CHUNK_FRAMES = 3;
    const NORMAL_CHUNK_FRAMES = 12;

    // Tracking across all chunks
    const allGeneratedLatents = [];
    let isFirstAudioChunk = true;
    let totalDecodedFrames = 0;
    let totalFlowLmTime = 0;
    let totalDecodeTime = 0;
    const arStartTime = performance.now();

    // Process each text chunk
    for (let chunkIdx = 0; chunkIdx < chunks.length; chunkIdx++) {
        if (!isGenerating) break;

        if (RESET_FLOW_STATE_EACH_CHUNK && chunkIdx > 0) {
            flowLmState = await buildVoiceConditionedState();
        }
        if (RESET_MIMI_STATE_EACH_CHUNK && chunkIdx > 0) {
            mimiState = initState(mimiDecoderSession, MIMI_DECODER_STATE_SHAPES);
        }

        const chunkText = chunks[chunkIdx];
        console.log(`Processing chunk ${chunkIdx + 1}/${chunks.length}: "${chunkText}"`);

        let isFirstAudioChunkOfTextChunk = true;

        // Tokenize this chunk
        const tokenIds = tokenizerProcessor.encodeIds(chunkText);
        console.log(`Chunk ${chunkIdx + 1} tokens:`, tokenIds.length);

        // Text conditioning for this chunk
        const textInput = new ort.Tensor('int64', BigInt64Array.from(tokenIds.map(x => BigInt(x))), [1, tokenIds.length]);
        const textCondResult = await textConditionerSession.run({ token_ids: textInput });
        let textEmb = textCondResult[textConditionerSession.outputNames[0]];

        if (textEmb.dims.length === 2) {
            textEmb = new ort.Tensor('float32', textEmb.data, [1, textEmb.dims[0], textEmb.dims[1]]);
        }

        const textCondInputs = {
            sequence: emptySeq,
            text_embeddings: textEmb,
            ...flowLmState
        };

        let condResult = await flowLmMainSession.run(textCondInputs);

        // Update state from text conditioning
        for (let i = 2; i < flowLmMainSession.outputNames.length; i++) {
            const outputName = flowLmMainSession.outputNames[i];
            if (outputName.startsWith('out_state_')) {
                const stateIdx = parseInt(outputName.replace('out_state_', ''));
                flowLmState[`state_${stateIdx}`] = condResult[outputName];
            }
        }

        // AR generation for this chunk
        const chunkLatents = [];
        let currentLatent = new ort.Tensor('float32', new Float32Array(32).fill(NaN), [1, 1, 32]);
        let chunkDecodedFrames = 0;
        const FRAMES_AFTER_EOS = 3;  // Match PyTorch behavior - generate extra frames after EOS
        let eosStep = null;

        let chunkEnded = false;
        let chunkGenTimeMs = 0;
        for (let step = 0; step < MAX_FRAMES; step++) {
            if (!isGenerating) break;

            // Yield every 4 steps to allow message processing (e.g., set_lsd)
            if (step > 0 && step % 4 === 0) {
                await new Promise(r => setTimeout(r, 0));
            }

            const arInputs = {
                sequence: currentLatent,
                text_embeddings: emptyTextEmb,
                ...flowLmState
            };

            const stepStart = performance.now();
            const arResult = await flowLmMainSession.run(arInputs);
            const stepElapsed = performance.now() - stepStart;
            chunkGenTimeMs += stepElapsed;

            const conditioning = arResult['conditioning'];
            const eosLogit = arResult['eos_logit'].data[0];
            const isEos = eosLogit > -4.0;

            // Track when EOS is first detected
            if (isEos && eosStep === null) {
                eosStep = step;
            }

            // Only stop after FRAMES_AFTER_EOS additional frames
            const shouldStop = eosStep !== null && step >= eosStep + FRAMES_AFTER_EOS;

            // Flow matching (LSD loop) - uses currentLSD which can be adjusted dynamically
            const TEMP = 0.7;
            const STD = Math.sqrt(TEMP);
            let xData = new Float32Array(32);
            for (let i = 0; i < 32; i++) {
                let u = 0, v = 0;
                while (u === 0) u = Math.random();
                while (v === 0) v = Math.random();
                xData[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v) * STD;
            }

            const lsdSteps = currentLSD;
            const dt = 1.0 / lsdSteps;

            for (let j = 0; j < lsdSteps; j++) {
                const flowInputs = {
                    c: conditioning,
                    s: stTensors[lsdSteps][j].s,
                    t: stTensors[lsdSteps][j].t,
                    x: new ort.Tensor('float32', xData, [1, 32])
                };

                const flowResult = await flowLmFlowSession.run(flowInputs);
                const v = flowResult['flow_dir'].data;

                for (let k = 0; k < 32; k++) {
                    xData[k] += v[k] * dt;
                }
            }

            totalFlowLmTime += stepElapsed;

            const latentData = xData;
            chunkLatents.push(new Float32Array(latentData));
            allGeneratedLatents.push(new Float32Array(latentData));

            // Update state
            currentLatent = new ort.Tensor('float32', latentData, [1, 1, 32]);
            for (let i = 2; i < flowLmMainSession.outputNames.length; i++) {
                const outputName = flowLmMainSession.outputNames[i];
                if (outputName.startsWith('out_state_')) {
                    const stateIdx = parseInt(outputName.replace('out_state_', ''));
                    flowLmState[`state_${stateIdx}`] = arResult[outputName];
                }
            }

            // Decode audio chunks
            const pending = chunkLatents.length - chunkDecodedFrames;
            let decodeSize = 0;

            if (shouldStop) {
                decodeSize = pending;
            } else if (isFirstAudioChunk && pending >= FIRST_CHUNK_FRAMES) {
                decodeSize = FIRST_CHUNK_FRAMES;
            } else if (pending >= NORMAL_CHUNK_FRAMES) {
                decodeSize = NORMAL_CHUNK_FRAMES;
            }

            if (decodeSize > 0) {
                const decodeLatents = new Float32Array(decodeSize * 32);
                for (let i = 0; i < decodeSize; i++) {
                    decodeLatents.set(chunkLatents[chunkDecodedFrames + i], i * 32);
                }

                const latentTensor = new ort.Tensor('float32', decodeLatents, [1, decodeSize, 32]);
                const decodeInputs = { latent: latentTensor, ...mimiState };

                const decStart = performance.now();
                const decodeResult = await mimiDecoderSession.run(decodeInputs);
                const decElapsed = performance.now() - decStart;
                totalDecodeTime += decElapsed;
                chunkGenTimeMs += decElapsed;
                const audioChunk = decodeResult[mimiDecoderSession.outputNames[0]].data;

                // Update MIMI state
                for (let i = 1; i < mimiDecoderSession.outputNames.length; i++) {
                    const outputName = mimiDecoderSession.outputNames[i];
                    const stateIdx = i - 1;
                    mimiState[`state_${stateIdx}`] = decodeResult[outputName];
                }

                chunkDecodedFrames += decodeSize;
                totalDecodedFrames += decodeSize;

                const audioFloat32 = new Float32Array(audioChunk);
                const isLastChunk = shouldStop && chunkIdx === chunks.length - 1;
                postMessage({
                    type: 'audio_chunk',
                    data: audioFloat32,
                    metrics: {
                        bbTime: 0,
                        decTime: 0,
                        chunkDuration: audioFloat32.length / SAMPLE_RATE,
                        genTimeSec: chunkGenTimeMs / 1000,
                        isFirst: isFirstAudioChunk,
                        isLast: isLastChunk,
                        chunkStart: isFirstAudioChunkOfTextChunk
                    }
                }, [audioFloat32.buffer]);

                isFirstAudioChunk = false;
                isFirstAudioChunkOfTextChunk = false;
                chunkGenTimeMs = 0;
            }

            if (shouldStop) {
                console.log(`Chunk ${chunkIdx + 1} EOS at step ${eosStep}, stopped at step ${step}, ${chunkLatents.length} frames`);
                chunkEnded = true;
                break;
            }
        }

        if (chunkEnded && isGenerating && chunkIdx < chunks.length - 1) {
            const gapSamples = Math.max(1, Math.floor(CHUNK_GAP_SEC * SAMPLE_RATE));
            const silence = new Float32Array(gapSamples);
            postMessage({
                type: 'audio_chunk',
                data: silence,
                metrics: {
                    bbTime: 0,
                    decTime: 0,
                    chunkDuration: gapSamples / SAMPLE_RATE,
                    isFirst: false,
                    isLast: false,
                    isSilence: true
                }
            }, [silence.buffer]);
        }
    }

    const totalTime = (performance.now() - arStartTime) / 1000;
    const audioSeconds = allGeneratedLatents.length * SAMPLES_PER_FRAME / SAMPLE_RATE;

    // RTFx based on actual generation time (flow LM + decoder), not including conditioning
    const genTime = (totalFlowLmTime + totalDecodeTime) / 1000;
    const rtfx = audioSeconds / genTime;

    console.log(`Generation complete: ${allGeneratedLatents.length} frames (${audioSeconds.toFixed(2)}s audio)`);
    console.log(`  Total time: ${totalTime.toFixed(2)}s`);
    console.log(`  Gen time: ${genTime.toFixed(2)}s, RTFx: ${rtfx.toFixed(2)}x`);
    console.log(`  Flow LM: ${(totalFlowLmTime / 1000).toFixed(2)}s (${(totalFlowLmTime / allGeneratedLatents.length).toFixed(1)}ms/step)`);
    console.log(`  Decoder: ${(totalDecodeTime / 1000).toFixed(2)}s`);

    postMessage({
        type: 'status',
        status: `Finished (RTFx: ${rtfx.toFixed(2)}x)`,
        state: 'idle',
        metrics: { rtfx, genTime, totalTime, audioDuration: audioSeconds }
    });
}

// Pre-allocated buffers for step counter updates (avoid GC pressure in hot loop)
const stepBuffers = {};

function updateStateSteps(state, increment) {
    // Update step counters in state dict - reuse buffers to avoid allocation
    const incBigInt = BigInt(increment);
    for (const key in state) {
        if (key.includes('step') && state[key]) {
            const tensor = state[key];
            if (tensor.data instanceof BigInt64Array) {
                // Reuse buffer if same size, otherwise create new one
                if (!stepBuffers[key] || stepBuffers[key].length !== tensor.data.length) {
                    stepBuffers[key] = new BigInt64Array(tensor.data.length);
                }
                const buf = stepBuffers[key];
                for (let i = 0; i < tensor.data.length; i++) {
                    buf[i] = tensor.data[i] + incBigInt;
                }
                state[key] = new ort.Tensor('int64', buf, tensor.dims);
            }
        }
    }
}
