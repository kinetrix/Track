# FaceTrack documentation

| What | Where |
|---|---|
| Design decisions (full) | `../DESIGN.md` |
| Build and run | `../README.md` |
| Sep 25 summary (simple) | `../FaceTrack_Project_Summary.docx` |
| Oct 9 changes: 512 input, one-go reads, 60 FPS capture | `FaceTrack_Recent_Changes.docx` (export of the online doc) |
| Online doc (live version) | https://claude.ai/code/artifact/91ba3ab5-4134-415f-b28b-d7f3f56106e6 |
| FigJam diagram: camera to screen, how it got 3x faster | https://www.figma.com/board/g2g4Co8gvQW5ZKZVpOUuom |
| Quality checks behind the Oct 9 changes | `../conversion/eval/` |

## Final numbers (iPhone 17, one face)

| | Before (Sep 25) | Now (Oct 9) |
|---|---|---|
| Time per frame | 75.9 ms | ~23 ms |
| Frames processed and tracked per second | 10 | 30 |
| Camera capture | 30 FPS | 60 FPS |
| Face detection rate | every 3rd processed frame | every 3rd processed frame (~10/s) |
| EdgeTAM input size | 1024 | 512 |

## Models

| Model | Licence | Parameters | Shipped size |
|---|---|---|---|
| YuNet (OpenCV Zoo, 2023mar) | MIT | 0.053M | 0.28 MB Core ML |
| EdgeTAM (Meta, CVPR 2025) | Apache-2.0 | 13.9M | 36 MB Core ML fp16 (4 graphs) |
