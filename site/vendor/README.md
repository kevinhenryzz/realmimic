# Vendored libraries

Unmodified files from the npm packages below, loaded as ES modules through the import map in
`index.html` (no bundler). Each directory holds the package's licence.

| Directory | Package | Version | Files | Licence |
|---|---|---|---|---|
| `three/` | [three](https://www.npmjs.com/package/three) | 0.186.0 | `build/three.module.js`, `build/three.core.js`, `examples/jsm/controls/OrbitControls.js`; `examples/jsm/postprocessing/` EffectComposer, RenderPass, UnrealBloomPass, ShaderPass, OutputPass, Pass, MaskPass; `examples/jsm/shaders/` CopyShader, LuminosityHighPassShader, OutputShader; `examples/jsm/loaders/GLTFLoader.js`, `examples/jsm/utils/` BufferGeometryUtils, SkeletonUtils, `examples/jsm/environments/RoomEnvironment.js` (as `addons/...`) | MIT |
| `mujoco/` | [@mujoco/mujoco](https://www.npmjs.com/package/@mujoco/mujoco) | 3.16.0 | `mujoco.js`, `mujoco.wasm` (single-threaded build) | Apache-2.0 |
| `meshopt/` | [meshoptimizer](https://www.npmjs.com/package/meshoptimizer) | 1.2.0 | `meshopt_decoder.mjs` (decodes `assets/robot/meshes.bin`) | MIT |
| `fonts/` | [@fontsource/barlow-condensed](https://www.npmjs.com/package/@fontsource/barlow-condensed), [@fontsource/inter](https://www.npmjs.com/package/@fontsource/inter), [@fontsource/jetbrains-mono](https://www.npmjs.com/package/@fontsource/jetbrains-mono) | 5.3.0 | `files/<family>-latin-<weight>-normal.woff2` (the weights the page uses) | OFL-1.1 (`<family>-LICENSE`) |
| `onnxruntime-web/` | [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) | 1.30.0 | `dist/ort.wasm.min.mjs`, `dist/ort-wasm-simd-threaded.mjs`, `dist/ort-wasm-simd-threaded.wasm` (WASM backend, run with one thread) | MIT |

npm tarball integrity (`npm pack <name>@<version>`):

- three 0.186.0: `sha512-cr/fIM2ddMSVbYVgkfD4jLJv7Fh/8ZTjvo+7gQeSVGUZHxpx9FDwoL5iC7hUz/LiRA8wMbqfnb90xKfm1/HHkQ==`
- @mujoco/mujoco 3.16.0: `sha512-mlMzM1pEkXsqH5O+DuIukahPEWJgZqBTXVszdtMeYBrEOXAOepwKZn4cz1rQ4BaqWzRe+GuOczr0CmOXdsW2Yw==`
- meshoptimizer 1.2.0: `sha512-davRZeIJbxJrE24cwQle7ZDsxjdk/OphNOV83oX+efQinyoHY9Jcyz3MHbaoG0qySZajldGztNZ1RN/T19PZsg==`
- @fontsource/barlow-condensed, @fontsource/inter, @fontsource/jetbrains-mono 5.3.0 (OFL-1.1)
- onnxruntime-web 1.30.0: `sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ==`

The MuJoCo and ONNX Runtime packages ship no licence file; `mujoco/LICENSE` and
`onnxruntime-web/LICENSE` are the repositories' licences at the same tags
(google-deepmind/mujoco `3.16.0`, microsoft/onnxruntime `v1.30.0`).

SHA-256 of the vendored files:

```
9052042d676cb0fdc1ddfefe193053f34b7ac0513a616fdac4535d49987812ea  three/three.module.js
9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6  three/three.core.js
3d79d07ecb686b4e5d93232eedab255331c1beef711e13164eaa1f68655a5f2b  three/addons/controls/OrbitControls.js
4e079a5886152d7e529a59aef644e968ab4d32c6a33ce016b36bf29b2eac26f7  three/addons/postprocessing/EffectComposer.js
7cd08eee9d5d6f5578beaddbdcbe9c384f6873810af27f22ab7db3ceeb127aa3  three/addons/postprocessing/MaskPass.js
02e4a261af34de71338185e9e87f0cbe5cba9115608d984363e1269dec1d2272  three/addons/postprocessing/OutputPass.js
444b409c235ead986893c472e720da1b779a56985c7d10b279c7944b52bd61c5  three/addons/postprocessing/Pass.js
817f6c3cdcd0fd41515d112359ea0532568eefb5aabd3b33903957ebca1b8a6a  three/addons/postprocessing/RenderPass.js
e2500a5913b26bbf5148ceaae644c6edcff06a18b01494ee37bf856353d2ab9d  three/addons/postprocessing/ShaderPass.js
ba8f2fcadfa6588384c9473498f974d81d120f02f0e63a0e59c265202a006b5a  three/addons/postprocessing/UnrealBloomPass.js
a33057d5ac91c43304c186ac0e8816e62bb2ed471d3a00ff3018dfd5c0389718  three/addons/shaders/CopyShader.js
5044f780b6e6cf863947f64c36fe1587132f7fbe395ada863cd1e5f0388dcf1e  three/addons/shaders/LuminosityHighPassShader.js
353479f77a8d7e2629d49ccac9fc2f5dbfdda5442e0adf867b00377a2fcb0cb2  three/addons/shaders/OutputShader.js
131c0f78c01d19368ae495caa65b3adaa10487810a36a05bb5901b769a35ac16  three/addons/loaders/GLTFLoader.js
9fb63427ce6641fa14fd0baff9cc4d1b5f9c3d85fd084bf2e90e803c44ec1797  three/addons/utils/BufferGeometryUtils.js
b1632a703206c3d830de9fcbe515696770d04b71a15ee6b50afa6d2c3298c86f  three/addons/utils/SkeletonUtils.js
55f466192cc84298755a424c5e040345006b2ee1455589b3b54126c2ea4123f4  three/addons/environments/RoomEnvironment.js
5d5d89ddd0857a6518c628fa502d42f99d6dd67ca240182f1068cd30a45465ac  mujoco/mujoco.js
e05ac96db7f45bd46e76fed6e3469bf9204329e093344d742a18bca2cc0e230e  mujoco/mujoco.wasm
215a93c696f442034a46fbb382958f753fda60e30490683aeea6b235fcbb2b66  fonts/barlow-condensed-latin-600-normal.woff2
3787a5a419171630e6890cfa47c4da067474d005cd0ff8dc11ec090fdc3ee2b8  fonts/barlow-condensed-latin-700-normal.woff2
2515494e8cc2ca07c86ea78766cfa104b796aa6ed5d66821c01d51dbb0b52bce  fonts/barlow-condensed-latin-800-normal.woff2
8909904ab6c872eb994093482a88a28eca2cd95912d7b6fecd72103b0dc07edc  fonts/inter-latin-400-normal.woff2
f3779f1efccc4bdcdf9c0a02ab95bf6bd092ed09c48c08cedc725889edd1d19f  fonts/inter-latin-500-normal.woff2
f9a06e79cd3a2a20951c0f0e28f66dd0e6d3fda73911d640a2125c8fcb78f21a  fonts/inter-latin-600-normal.woff2
14425ba9c695763c1547f48a206b7aa60350a33ae23de09f0407877f3fcd89eb  fonts/jetbrains-mono-latin-400-normal.woff2
400c6bfda18d5d14acad1c15d6dcb9f8e13c015e7286317e0b9a482539bef147  fonts/jetbrains-mono-latin-600-normal.woff2
cb08bf53ad8ad9693d8bb759b2dbade350eb38bcce63d005385e225b564c5f6a  meshopt/meshopt_decoder.mjs
219e6a1fc8a9938268d18efca3c91d310bd2f4a59bbd13744df5b2b7fc6cee3b  onnxruntime-web/ort.wasm.min.mjs
e13f7f94fc51b4ca72b12faeb1ee95f4ace6dfbc8939bc718aabdc0a27c4299b  onnxruntime-web/ort-wasm-simd-threaded.mjs
3398c10d07d229bd91b364548e130e0e51a8e5704b88c7c083ebbeb78842dee2  onnxruntime-web/ort-wasm-simd-threaded.wasm
```

## Other bundled assets

- `assets/robot/`: the Unitree G1 model from [mjlab](https://github.com/mujocolab/mjlab) (Apache-2.0),
  based on Unitree Robotics' G1 description (BSD-3-Clause), compiled with MuJoCo 3.16.0
  (`tools/export_web_scene.py`); visual meshes decimated.
- `assets/robot/variants/g1_brainco/`: the BrainCo-hands variant's physics (`scene.mjb`, from realmimic's
  RobotSpec) and hand meshes from Unitree's [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
  `robots/g1_with_brainco_hand` at commit 15ebe53 (BSD-3-Clause), posed open and placed at the measured mount.
- `assets/policy/`: the realmimic policy (`tools/export_web_policy.py`). Training data includes Motion
  Data by Bones Studio (https://bones.studio/). Use of the underlying dataset is subject to the BONES
  Motion Capture Dataset License Agreement.
- `assets/motions/`: motions authored for the page, and clips retargeted to the G1 from the CMU Graphics Lab
  Motion Capture Database (mocap.cs.cmu.edu; created with funding from NSF EIA-0196217) and the 100STYLE
  dataset (Mason, Starke, Komura 2022; CC BY 4.0); `index.json` gives each clip's source, URL and licence.
- `assets/hero/`: a video rendered from the realmimic policy's rollouts on training clips (pixels only; Motion
  Data by Bones Studio, https://bones.studio/).

`tools/fetch_site_vendor.sh` also writes `mujoco/mujoco.wasm.gz` and `onnxruntime-web/ort-wasm-simd-threaded.wasm.gz`
(`gzip -9 -n` of the checked files above); the page downloads those and inflates them in the browser.
