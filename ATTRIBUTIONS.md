# Third-Party Attributions

This document lists all third-party libraries, models, and frameworks used in this project, along with their licenses and required attribution text.

---

## 1. Neural Network Inference

### ONNX Runtime (Android)

- **Version:** 1.29.0
- **Module:** `com.microsoft.onnxruntime:onnxruntime-android`
- **URL:** https://github.com/microsoft/onnxruntime
- **License:** MIT License

**Attribution:**

```
This project uses ONNX Runtime (https://github.com/microsoft/onnxruntime), an inference engine for running machine learning models. ONNX Runtime is licensed under the MIT License.

Copyright (c) Microsoft Corporation

All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### XNNPACK

- **Version:** Bundled with ONNX Runtime 1.29.0
- **Module:** `com.microsoft.onnxruntime:onnxruntime-android` (via `addXnnpack()`)
- **URL:** https://github.com/google/XNNPACK
- **License:** BSD-style (3-clause)

**What it is:** XNNPACK is a highly optimized C library of floating-point neural network operators for ARM, x86, WebAssembly, and RISC-V. It provides low-level performance primitives — convolution, matrix multiplication, activation, pooling — hand-tuned with SIMD intrinsics (NEON on ARM, AVX on x86) to maximize throughput. In this project it is enabled via `sessionOptions.addXnnpack(emptyMap())`, which routes ONNX Runtime operators to XNNPACK's NEON-accelerated kernels instead of naive reference implementations.

**Lineage:** XNNPACK originated from Meta's QNNPACK (Quantized Neural Network PACKage), open-sourced in 2018 for mobile quantized inference. Google extended it into full floating-point XNNPACK in 2019 and now maintains it.

**Attribution:**

```
This project uses XNNPACK (https://github.com/google/XNNPACK), a highly optimized floating-point neural network inference library for ARM/x86/WebAssembly/RISC-V, provided by ONNX Runtime. XNNPACK originated from Meta's QNNPACK and is now maintained by Google.

BSD License

For XNNPACK software

Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

Copyright 2019–2025 Google LLC

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of Facebook, Meta, nor Google, nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## 2. Computer Vision

### OpenCV

- **Version:** 5.0.0.1
- **Module:** `org.opencv:opencv`
- **URL:** https://github.com/opencv/opencv
- **License:** Apache License 2.0

**Attribution:**

```
This project uses OpenCV (https://github.com/opencv/opencv), an open-source computer vision and machine learning library. OpenCV is licensed under the Apache License 2.0.

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction,
and distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by
the copyright owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all
other entities that control, are controlled by, or are under common
control with that entity. For the purposes of this definition,
"control" means (i) the power, direct or indirect, to cause the
direction or management of such entity, whether by contract or
otherwise, or (ii) ownership of fifty percent (50%) or more of the
outstanding shares, or (iii) beneficial ownership of such entity.

"You" (or "Your") shall mean an individual or Legal Entity
exercising permissions granted by this License.

"Source" form shall mean the preferred form for making modifications,
including but not limited to software source code, documentation
source, and configuration files.

"Object" form shall mean any form resulting from mechanical
transformation or translation of a Source form, including but
not limited to compiled object code, generated documentation,
and conversions to other media types.

"Work" shall mean the work of authorship, whether in Source or
Object form, made available under the License, as indicated by a
copyright notice that is included in or attached to the work
(an example is provided in the Appendix below).

"Derivative Works" shall mean any work, whether in Source or Object
form, that is based on (or derived from) the Work and for which the
editorial revisions, annotations, elaborations, or other modifications
represent, as a whole, an original work of authorship. For the purposes
of this License, Derivative Works shall not include works that remain
separable from, or merely link (or bind by name) to the interfaces of,
the Work and Derivative Works thereof.

"Contribution" shall mean any work of authorship, including
the original version of the Work and any modifications or additions
to that Work or Derivative Works thereof, that is intentionally
submitted to the Licensor for inclusion in the Work by the copyright owner
or by an individual or Legal Entity authorized to submit on behalf of
the copyright owner. For the purposes of this definition, "submitted"
means any form of electronic, verbal, or written communication sent
to the Licensor or its representatives, including but not limited to
communication on electronic mailing lists, source code control systems,
and issue tracking systems that are managed by, or on behalf of, the
Licensor for the purpose of discussing and improving the Work, but
excluding communication that is conspicuously marked or otherwise
designated in writing by the copyright owner as "Not a Contribution."

"Contributor" shall mean Licensor and any individual or Legal Entity
on behalf of whom a Contribution has been received by Licensor and
subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
this License, each Contributor hereby grants to You a perpetual,
worldwide, non-exclusive, no-charge, royalty-free, irrevocable
copyright license to reproduce, prepare Derivative Works of,
publicly display, publicly perform, sublicense, and distribute the
Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
this License, each Contributor hereby grants to You a perpetual,
worldwide, non-exclusive, no-charge, royalty-free, irrevocable
(except as stated in this section) patent license to make, have made,
use, offer to sell, sell, import, and otherwise transfer the Work,
where such license applies only to those patent claims licensable
by such Contributor that are necessarily infringed by their
Contribution(s) alone or by combination of their Contribution(s)
with the Work to which such Contribution(s) was submitted. If You
institute patent litigation against any entity (including a
cross-claim or counterclaim in a lawsuit) alleging that the Work
or a Contribution incorporated within the Work constitutes direct
or contributory patent infringement, then any patent licenses
granted to You under this License for that Work shall terminate
as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
Work or Derivative Works thereof in any medium, with or without
modifications, and in Source or Object form, provided that You
meet the following conditions:

(a) You must give any other recipients of the Work or
Derivative Works a copy of this License; and

(b) You must cause any modified files to carry prominent notices
stating that You changed the files; and

(c) You must retain, in the Source form of any Derivative Works
that You distribute, all copyright, patent, trademark, and
attribution notices from the Source form of the Work,
excluding those notices that do not pertain to any part of
the Derivative Works; and

(d) If the Work is part of a larger work that includes this
License, You may include the license in that larger work, but
You must also provide a copy of this License with that larger work.

5. Submission of Contributions. Unless You explicitly state otherwise,
any Contribution intentionally submitted for inclusion in the Work
by You to the Licensor shall be under the terms and conditions of
this License, without any additional terms or conditions.
Notwithstanding the above, nothing herein shall supersede or modify
the terms of any separate license agreement you may have executed
with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
names, trademarks, service marks, or product names of the Licensor,
except as required for reasonable and customary use in describing the
origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
agreed to in writing, Licensor provides the Work (and each
Contributor provides its Contributions) on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied, including, without limitation, any warranties or conditions
of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
PARTICULAR PURPOSE. You are solely responsible for determining the
appropriateness of using or redistributing the Work and assume any
risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
whether in tort (including negligence), contract, or otherwise,
unless required by applicable law (such as deliberate and grossly
negligent acts) or agreed to in writing, shall any Contributor be
liable to You for damages, including any direct, indirect, special,
incidental, or consequential damages of any character arising as a
result of this License or out of the use or inability to use the
Work (including but not limited to damages for loss of goodwill,
work stoppage, computer failure or malfunction, or any and all
other commercial damages or losses), even if such Contributor
has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
the Work or Derivative Works thereof, You may choose to offer,
and charge a fee for, acceptance of support, warranty, indemnity,
or other liability obligations and/or rights consistent with this
License. However, in accepting such support, You may only offer
or accept such support on behalf of Yourself; any support offered
by any other party is at your own risk.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

To apply the Apache License to your work, attach the following
boilerplate notice, with the fields enclosed by brackets "[]"
replaced with your own identifying information. (Don't include
the brackets!) The text should be enclosed in an appropriate
comment syntax for the file format. We also recommend that a
file or class name and description of purpose be included on the
same "printed page" as the copyright notice for easier
identification within third-party archives.

Copyright [yyyy] [name of copyright owner]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 3. Android Framework & UI

### AndroidX CameraX

- **Version:** 1.6.1
- **Modules:** `androidx.camera:camera-core`, `androidx.camera:camera-camera2`, `androidx.camera:camera-lifecycle`, `androidx.camera:camera-view`
- **URL:** https://developer.android.com/training/camerax
- **License:** Apache License 2.0

**Attribution:**

```
This project uses AndroidX CameraX (https://developer.android.com/training/camerax), a jetpack library that provides a simple, powerful API for building camera applications on Android. CameraX is licensed under the Apache License 2.0.

Copyright 2019 The Android Open Source Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Jetpack Compose & AndroidX

- **Compose BOM:** 2026.06.01
- **Lifecycle:** 2.11.0
- **Activity:** 1.13.0
- **Navigation:** 2.9.8
- **Core KTX:** 1.19.0
- **EXIF Interface:** 1.4.2
- **URL:** https://developer.android.com/jetpack
- **License:** Apache License 2.0

**Attribution:**

```
This project uses Jetpack Compose and AndroidX libraries (https://developer.android.com/jetpack) developed by Google and the Android Open Source Project. These libraries are licensed under the Apache License 2.0.

Copyright 2019 The Android Open Source Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Koin

- **Version:** 4.2.2 (BOM)
- **Modules:** `io.insert-koin:koin-core`, `io.insert-koin:koin-android`, `io.insert-koin:koin-androidx-compose`
- **URL:** https://insert-koin.io
- **License:** Apache License 2.0

**Attribution:**

```
This project uses Koin (https://insert-koin.io), a lightweight dependency injection framework for Kotlin. Koin is licensed under the Apache License 2.0.

Copyright 2017 - 2025 Koin Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 4. ONNX Models

### LCNet100 + BiFPN (Heatmap Corner Detector)

- **Source:** DocsaidLab
- **Size:** ~5 MB
- **Purpose:** Predicts 4 heatmaps (one per document corner) via LCNet backbone + BiFPN feature pyramid. Centroid extraction yields sub-pixel corner accuracy.
- **URL:** https://github.com/DocsaidLab/DocAligner
- **License:** Apache 2.0

**Attribution:**

```
This project uses the LCNet100 + BiFPN heatmap corner detection model from DocsaidLab (https://github.com/DocsaidLab/DocAligner), licensed under Apache 2.0. The model predicts document corner locations via heatmap regression and is used as a fallback detector when classical OpenCV methods fail.
```

### LCNet (Corner Keypoint Detector)

- **Source:** DocsaidLab
- **Size:** ~5 MB
- **Purpose:** Predicts document corner keypoints directly. Kept in the codebase for comparison but disabled in release builds due to larger prediction errors.
- **URL:** https://github.com/DocsaidLab/DocAligner
- **License:** Apache 2.0

**Attribution:**

```
This project includes the LCNet corner keypoint detection model from DocsaidLab (https://github.com/DocsaidLab/DocAligner), licensed under Apache 2.0. The model predicts document corner coordinates directly. It is intentionally disabled in release builds due to unreliable predictions.
```

### DeepLabV3-MobileNetV3 (Segmentation Detector)

- **Source:** mukund-ks / DeepLab community
- **Size:** ~42 MB
- **Purpose:** Semantic segmentation model that produces a pixel-level document mask. The mask is then processed through morphological cleaning, largest-component filtering, and quad extraction.
- **URL:** https://github.com/mukund-ks/DeepLabV3-Segmentation
- **License:** Apache 2.0

**Attribution:**

```
This project uses a DeepLabV3 semantic segmentation model with MobileNetV3 backbone from mukund-ks (https://github.com/mukund-ks/DeepLabV3-Segmentation), licensed under Apache 2.0. The model produces a pixel-level document mask for perspective correction.
```

---

## 5. Summary Table

| Library | Version | License | Risk |
|---|---|---|---|
| ONNX Runtime | 1.29.0 | MIT | ✅ Clear |
| XNNPACK | Bundled | BSD-3 | ✅ Clear |
| OpenCV | 5.0.0.1 | Apache 2.0 | ✅ Clear |
| CameraX | 1.6.1 | Apache 2.0 | ✅ Clear |
| AndroidX / Compose | Latest | Apache 2.0 | ✅ Clear |
| Koin | 4.2.2 | Apache 2.0 | ✅ Clear |
| LCNet Heatmap (DocsaidLab) | ~5 MB | Apache 2.0 | ✅ Clear |
| LCNet Keypoint (DocsaidLab) | ~5 MB | Apache 2.0 | ✅ Clear |
| DeepLabV3-MobileNetV3 | ~42 MB | Apache 2.0 | ✅ Clear |

---

## 6. Notes

- **XNNPACK** is not a separate dependency — it is bundled inside ONNX Runtime and activated via `sessionOptions.addXnnpack(emptyMap())`. The attribution is included here because the BSD license requires retaining the copyright notice.
- **DocsaidLab models** are licensed under Apache 2.0 via the DocAligner repository (https://github.com/DocsaidLab/DocAligner), and the **DeepLabV3 segmentation model** is licensed under Apache 2.0 via https://github.com/mukund-ks/DeepLabV3-Segmentation. All dependencies in this project are permissively licensed (MIT, BSD-3, or Apache 2.0).
- All Apache 2.0 and MIT-licensed libraries require retaining the copyright notice and license text in redistributions. This file satisfies that requirement for binary distribution.
