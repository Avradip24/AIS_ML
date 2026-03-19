# FIUS Classification - Final Result Tables

Generated: 2026-03-19T20:59:49.568415

## Summary Table

| Metric Type | Metric Name | Value | Unit | Source |
|-------------|-------------|-------|------|--------|
| segment_level | validation_accuracy | .3f | % | training_history.json |
| segment_level | macro_f1 | .3f |  | training_history.json |
| segment_level | validation_loss | .3f |  | training_history.json |
| file_level | file_accuracy | .3f | % | file_level_results.json |
| file_level | valid_files | .3f | count | file_level_results.json |
| file_level_per_class | person_accuracy | .3f | % | file_level_results.json |
| file_level_per_class | plant_accuracy | .3f | % | file_level_results.json |
| file_level_per_class | backpack_accuracy | .3f | % | file_level_results.json |
| file_level_per_class | wall_accuracy | .3f | % | file_level_results.json |
| file_level_per_class | chair_accuracy | .3f | % | file_level_results.json |
| latency | total_inference_time | .3f | ms | latency_profile.json |
| latency | preprocessing_time | .3f | ms | latency_profile.json |
| latency | forward_pass_time | .3f | ms | latency_profile.json |
| latency | aggregation_time | .3f | ms | latency_profile.json |
| latency | per_pulse_forward_time | .3f | ms | latency_profile.json |

## Segment-Level Metrics

- Validation Accuracy: 78.56%
- Macro F1: 0.7973
- Validation Loss: 0.8101
- Source: training_history.json

## File-Level Metrics

- File Accuracy: 100.00%
- Valid Files: 6
- Total Files: 6
- Error Files: 0

### Per-Class File Accuracy

| Class | Accuracy |
|-------|----------|
| backpack | 100.00% |
| chair | 100.00% |
| person | 100.00% |
| plant | 100.00% |
| wall | 100.00% |

## Latency Metrics

- Total Inference Time: 1.54 ms
- Preprocessing Time: 0.00 ms
- Forward Pass Time: 1.54 ms
- Aggregation Time: 0.00 ms
- Per-Pulse Forward Time: 1.544 ms
- Files Profiled: 1
- Source: latency_profile.json

✅ **Meets AIS <10ms latency requirement**

