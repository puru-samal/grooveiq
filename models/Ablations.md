# 🥁 GrooveIQ Ablation Study Tracking Sheet

| Exp ID | Encoder | Decoder     | Param Target | # Params (actual) | hit_acc ↑ | hit_ppv ↑ | hit_tpr ↑ | hit_f1 ↑ | velocity_mse ↓ | offset_mse ↓ | offset_tightness ↑ | offset_ahead ↑ | offset_behind ↑ | Notes |
| ------ | ------- | ----------- | ------------ | ----------------- | --------- | --------- | --------- | -------- | -------------- | ------------ | ------------------ | -------------- | --------------- | ----- |
| E1     | MLP     | MLP         | ~2.5M        |                   |           |           |           |          |                |              |                    |                |                 |       |
| E2     | MLP     | GRU         | ~2.5M        |                   |           |           |           |          |                |              |
| E3     | MLP     | Conv        | ~2.5M        |                   |           |           |           |          |                |              |
| E4     | MLP     | Transformer | ~2.5M        |                   |           |           |           |          |                |              |
| E5     | Conv    | MLP         | ~2.5M        |                   |           |           |           |          |                |              |
| E6     | Conv    | GRU         | ~2.5M        |                   |           |           |           |          |                |              |
| E7     | Conv    | Conv        | ~2.5M        |                   |           |           |           |          |                |              |
| E8     | Conv    | Transformer | ~2.5M        |                   |           |           |           |          |                |              |
| E9     | Axial   | MLP         | ~2.5M        |                   |           |           |           |          |                |              |
| E10    | Axial   | GRU         | ~2.5M        |                   |           |           |           |          |                |              |
| E11    | Axial   | Conv        | ~2.5M        |                   |           |           |           |          |                |              |
| E12    | Axial   | Transformer | ~2.5M        |                   |           |           |           |          |                |              |
