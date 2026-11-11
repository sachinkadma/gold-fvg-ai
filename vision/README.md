# Vision Subsystem: Computer Vision Setup Detection

## Model Overview
- **Weights**: `best_v2_4class.pt`, `best.pt`
- **Classes**: Bullish FVG, Bearish FVG, Bullish MSS, Bearish MSS

## Crucial Audit Finding: Metric Interpretation
The reported **95.6% mAP@50** accuracy evaluates **bounding box localization agreement** against synthetic rule-generated labels on chart images. 

> [!WARNING]
> **95.6% measures visual agreement with rule-generated chart annotations, NOT a predictive trading edge.**
> 
> A computer vision model replicating rule-based boxes cannot overcome execution friction ($0.35 spread, $0.10 slippage, $7/lot commission) if the underlying mechanical rules lack positive expectancy. Computer vision serves purely as an annotation and rendering utility, not as an alpha generator.
