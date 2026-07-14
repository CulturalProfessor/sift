import { readFileSync } from 'fs';
import { CLIPTokenizer } from '../src/native/tokenizer';

const merges = readFileSync(
  'android/app/src/main/assets/bpe_merges.txt',
  'utf-8',
);
const tok = new CLIPTokenizer(merges);

const expected: Record<string, number[]> = {
  dog: [49406, 1929, 49407],
  'a photo of a dog': [49406, 320, 1125, 539, 320, 1929, 49407],
  'sunset over the beach': [49406, 3424, 962, 518, 2117, 49407],
  'Hello World 123': [49406, 3306, 1002, 272, 273, 274, 49407],
  'a cat & a dog': [49406, 320, 2368, 261, 320, 1929, 49407],
};

let allOk = true;
for (const [text, exp] of Object.entries(expected)) {
  const ids = tok.tokenize(text).filter((v, i) => i === 0 || v !== 0); // drop padding
  const got = ids.slice(0, exp.length);
  const ok = JSON.stringify(got) === JSON.stringify(exp);
  allOk = allOk && ok;
  console.log(`${ok ? 'OK ' : 'FAIL'}  ${JSON.stringify(text)} -> ${JSON.stringify(got)}`);
  if (!ok) console.log(`      expected ${JSON.stringify(exp)}`);
}
console.log(allOk ? '\nALL MATCH ✓' : '\nMISMATCH ✗');
process.exit(allOk ? 0 : 1);
