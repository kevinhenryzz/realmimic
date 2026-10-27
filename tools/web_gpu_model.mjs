// The RTX 4090 model on the page (assets/gpu/rtx4090.glb) from the Sketchfab download:
// "GeForce RTX 4090 Founders Edition" by exéla (https://skfb.ly/oyBLN), CC BY-NC 4.0.
// Changes: brand lettering painted out (side "GEFORCE RTX", the marks on the bracket; "RTX 4090" kept, turned
// the right way up for this view), unused
// UV sets and tangents dropped, geometry simplified, quantized and meshopt-compressed, textures to WebP.
// Copy into a directory with @gltf-transform/{core,extensions,functions}@4.1.1, meshoptimizer@0.22.0 and sharp@0.33.5
// installed (ES modules resolve packages next to the script) and run it there:
//   node web_gpu_model.mjs <download.glb> <repo>/site/assets/gpu/rtx4090.glb
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { dedup, prune, quantize, simplify, textureCompress, weld, meshopt } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';

const [src, out] = process.argv.slice(2);
// Painted out, in pixels of the 1024 x 1024 main-material maps: [x0, y0, x1, y1].
const SIDE_LETTERING = [40, 145, 68, 268];
const BRACKET_MARKS = [500, 580, 562, 668];
// "RTX 4090" on the backplate: upside down with the card lying connector-down, as the page shows it.
const BACKPLATE_TEXT = [900, 66, 923, 181];

// Fill rectangles with the median colour of a 6-pixel ring around each, so the patch matches the surface.
async function paint(bytes, rects, flat) {
  const img = sharp(bytes);
  const { width, height, channels } = await img.metadata();
  const raw = await img.raw().toBuffer();
  for (const [x0, y0, x1, y1] of rects) {
    const ring = [];
    for (let y = y0 - 6; y < y1 + 6; y++) for (let x = x0 - 6; x < x1 + 6; x++) {
      if (x < 0 || y < 0 || x >= width || y >= height || (x >= x0 && x < x1 && y >= y0 && y < y1)) continue;
      ring.push(raw.subarray((y * width + x) * channels, (y * width + x + 1) * channels));
    }
    const fill = flat ?? [...Array(channels).keys()].map((c) => ring.map((p) => p[c]).sort((a, b) => a - b)[ring.length >> 1]);
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) raw.set(fill.slice(0, channels), (y * width + x) * channels);
  }
  return sharp(raw, { raw: { width, height, channels } }).png().toBuffer();
}

// Turn a rectangle by 180 degrees; on a tangent-space normal map both slopes (red, green) change sign with it.
async function turn(bytes, [x0, y0, x1, y1], normal = false) {
  const img = sharp(bytes);
  const { width, height, channels } = await img.metadata();
  const raw = await img.raw().toBuffer();
  const src = Buffer.from(raw);
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const from = ((y0 + y1 - 1 - y) * width + (x0 + x1 - 1 - x)) * channels, to = (y * width + x) * channels;
    src.copy(raw, to, from, from + channels);
    if (normal) { raw[to] = 255 - raw[to]; raw[to + 1] = 255 - raw[to + 1]; }
  }
  return sharp(raw, { raw: { width, height, channels } }).png().toBuffer();
}

await MeshoptEncoder.ready;
await MeshoptSimplifier.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.encoder': MeshoptEncoder });
const doc = await io.read(src);
const root = doc.getRoot();

const main = root.listMaterials().find((m) => m.getName() === 'main_material');
const edits = [
  [main.getBaseColorTexture(), [SIDE_LETTERING, BRACKET_MARKS]],
  [main.getMetallicRoughnessTexture(), [SIDE_LETTERING, BRACKET_MARKS]],
  [main.getEmissiveTexture(), [SIDE_LETTERING], [0, 0, 0, 255]], // the lettering glows: emissive off there
  [main.getNormalTexture(), [BRACKET_MARKS], [128, 128, 255, 255]], // flat
];
for (const [tex, rects, flat] of edits) tex.setImage(await paint(tex.getImage(), rects, flat));
main.getBaseColorTexture().setImage(await turn(main.getBaseColorTexture().getImage(), BACKPLATE_TEXT));
main.getNormalTexture().setImage(await turn(main.getNormalTexture().getImage(), BACKPLATE_TEXT, true));

// Every material samples TEXCOORD_0; three.js shades normal maps without stored tangents.
for (const mesh of root.listMeshes()) for (const prim of mesh.listPrimitives()) {
  for (const name of ['TEXCOORD_1', 'TEXCOORD_2', 'TANGENT']) {
    const acc = prim.getAttribute(name);
    if (acc) { prim.setAttribute(name, null); }
  }
}
await doc.transform(
  dedup(), prune(), weld(),
  simplify({ simplifier: MeshoptSimplifier, ratio: 0.5, error: 0.0005 }),
  textureCompress({ encoder: sharp, targetFormat: 'webp', quality: 82, resize: [1024, 1024] }),
  quantize(),
  meshopt({ encoder: MeshoptEncoder, level: 'high' }),
);
await io.write(out, doc);
const tris = root.listMeshes().flatMap((m) => m.listPrimitives()).reduce((s, p) => s + (p.getIndices()?.getCount() ?? 0) / 3, 0);
console.log(`${out}: ${tris} triangles`);
