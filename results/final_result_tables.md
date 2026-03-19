# FIUS Classification - Final Result Tables

Generated: 2026-03-18T12:32:56.949935

## Summary Table

| Metric Type | Metric Name | Value | Unit | Source |
|-------------|-------------|-------|------|--------|
| segment_level | validation_accuracy | .3f | % | training_history.json |
| segment_level | macro_f1 | .3f |  | training_history.json |
| segment_level | validation_loss | .3f |  | training_history.json |
| file_level | file_accuracy | .3f | % | file_level_results.json |
| file_level | valid_files | .3f | count | file_level_results.json |
| latency | total_inference_time | .3f | ms | latency_profile.json |
| latency | preprocessing_time | .3f | ms | latency_profile.json |
| latency | forward_pass_time | .3f | ms | latency_profile.json |
| latency | aggregation_time | .3f | ms | latency_profile.json |
| latency | per_pulse_forward_time | .3f | ms | latency_profile.json |

## Segment-Level Metrics

- Validation Accuracy: 77.44%
- Macro F1: 0.7725
- Validation Loss: 0.8403
- Source: training_history.json

## File-Level Metrics

- File Accuracy: 0.00%
- Valid Files: 0
- Total Files: 3
- Error Files: 3

## Latency Metrics

- Total Inference Time: 170.33 ms
- Preprocessing Time: 162.61 ms
- Forward Pass Time: 7.52 ms
- Aggregation Time: 0.20 ms
- Per-Pulse Forward Time: 0.150 ms
- Files Profiled: 5
- Source: latency_profile.json

❌ **Exceeds AIS 10ms latency requirement**

