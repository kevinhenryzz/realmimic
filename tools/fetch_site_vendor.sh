#!/usr/bin/env bash
# Fetch the website's pinned JavaScript / WASM libraries from the npm registry into site/vendor/ and check
# them against the SHA-256 list in site/vendor/README.md (the files are not committed).
#   tools/fetch_site_vendor.sh
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fetch() {  # registry path, then "tarball-path:vendor-path" pairs
  local url=$1; shift
  curl -sfL "https://registry.npmjs.org/$url" -o "$tmp/pkg.tgz"
  rm -rf "$tmp/package" && tar -xzf "$tmp/pkg.tgz" -C "$tmp"
  for pair in "$@"; do
    mkdir -p "site/vendor/$(dirname "${pair#*:}")"
    cp "$tmp/package/${pair%%:*}" "site/vendor/${pair#*:}"
  done
}
addons=(controls/OrbitControls.js postprocessing/{EffectComposer,RenderPass,UnrealBloomPass,ShaderPass,OutputPass,Pass,MaskPass}.js
  shaders/{CopyShader,LuminosityHighPassShader,OutputShader}.js loaders/GLTFLoader.js
  utils/{BufferGeometryUtils,SkeletonUtils}.js environments/RoomEnvironment.js)
pairs=()
for a in "${addons[@]}"; do pairs+=("examples/jsm/$a:three/addons/$a"); done
fetch three/-/three-0.186.0.tgz build/three.module.js:three/three.module.js build/three.core.js:three/three.core.js "${pairs[@]}"
fetch @mujoco/mujoco/-/mujoco-3.16.0.tgz mujoco.js:mujoco/mujoco.js mujoco.wasm:mujoco/mujoco.wasm
fetch onnxruntime-web/-/onnxruntime-web-1.30.0.tgz dist/ort.wasm.min.mjs:onnxruntime-web/ort.wasm.min.mjs \
  dist/ort-wasm-simd-threaded.mjs:onnxruntime-web/ort-wasm-simd-threaded.mjs \
  dist/ort-wasm-simd-threaded.wasm:onnxruntime-web/ort-wasm-simd-threaded.wasm
fetch meshoptimizer/-/meshoptimizer-1.2.0.tgz meshopt_decoder.mjs:meshopt/meshopt_decoder.mjs
# Fonts (OFL-1.1): latin subsets of the weights the page uses.
fetch @fontsource/barlow-condensed/-/barlow-condensed-5.3.0.tgz LICENSE:fonts/barlow-condensed-LICENSE \
  files/barlow-condensed-latin-{600,700,800}-normal.woff2:fonts/
fetch @fontsource/inter/-/inter-5.3.0.tgz LICENSE:fonts/inter-LICENSE files/inter-latin-{400,500,600}-normal.woff2:fonts/
fetch @fontsource/jetbrains-mono/-/jetbrains-mono-5.3.0.tgz LICENSE:fonts/jetbrains-mono-LICENSE \
  files/jetbrains-mono-latin-{400,600}-normal.woff2:fonts/
(cd site/vendor && sed -n '/^```$/,/^```$/p' README.md | grep -E '^[0-9a-f]{64}  ' | sha256sum -c --quiet)
# Gzipped copies of the two WASM binaries: the page fetches these and inflates them itself (about 6 MB
# instead of 25 MB, whatever compression the host applies).
for f in mujoco/mujoco.wasm onnxruntime-web/ort-wasm-simd-threaded.wasm; do gzip -9 -n -c "site/vendor/$f" > "site/vendor/$f.gz"; done
echo "site/vendor: all files fetched and verified"
