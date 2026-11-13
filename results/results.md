# Evaluation results (split: test)

## Summary, all images each method ran on
| method | macro_f1 | defect_recall | false_reject_rate | image_accuracy | exact_match | mean_time_s | total_cost | n_images |
|---|---|---|---|---|---|---|---|---|
| dip | 0.853 | 0.961 | 0.024 | 0.966 | 0.795 | 0.027 | 0.000 | 352 |
| cnn | 0.995 | 0.996 | 0.000 | 0.997 | 0.991 | 0.033 | 0.000 | 352 |
| vlm | 0.662 | 1.000 | 0.290 | 0.898 | 0.514 | 2.307 | 0.036 | 352 |

## Summary, common subset (n=352)
| method | macro_f1 | defect_recall | false_reject_rate | image_accuracy | exact_match | mean_time_s | total_cost | n_images |
|---|---|---|---|---|---|---|---|---|
| dip | 0.853 | 0.961 | 0.024 | 0.966 | 0.795 | 0.027 | 0.000 | 352 |
| cnn | 0.995 | 0.996 | 0.000 | 0.997 | 0.991 | 0.033 | 0.000 | 352 |
| vlm | 0.662 | 1.000 | 0.290 | 0.898 | 0.514 | 2.307 | 0.036 | 352 |

## Per-defect metrics, all images
| method | defect | precision | recall | f1 | support | n_images |
|---|---|---|---|---|---|---|
| dip | blur | 0.872 | 0.788 | 0.828 | 52 | 352 |
| dip | noise | 0.979 | 1.000 | 0.989 | 47 | 352 |
| dip | dark | 0.804 | 0.837 | 0.820 | 49 | 352 |
| dip | non_white_bg | 0.969 | 1.000 | 0.984 | 63 | 352 |
| dip | bad_framing | 0.735 | 0.926 | 0.820 | 54 | 352 |
| dip | cropped | 0.647 | 0.702 | 0.673 | 47 | 352 |
| cnn | blur | 1.000 | 1.000 | 1.000 | 52 | 352 |
| cnn | noise | 1.000 | 1.000 | 1.000 | 47 | 352 |
| cnn | dark | 0.980 | 1.000 | 0.990 | 49 | 352 |
| cnn | non_white_bg | 1.000 | 1.000 | 1.000 | 63 | 352 |
| cnn | bad_framing | 1.000 | 1.000 | 1.000 | 54 | 352 |
| cnn | cropped | 0.979 | 0.979 | 0.979 | 47 | 352 |
| vlm | blur | 0.591 | 1.000 | 0.743 | 52 | 352 |
| vlm | noise | 0.653 | 1.000 | 0.790 | 47 | 352 |
| vlm | dark | 0.875 | 0.571 | 0.691 | 49 | 352 |
| vlm | non_white_bg | 0.516 | 1.000 | 0.681 | 63 | 352 |
| vlm | bad_framing | 0.412 | 1.000 | 0.584 | 54 | 352 |
| vlm | cropped | 0.739 | 0.362 | 0.486 | 47 | 352 |

## Per-defect metrics, common subset (n=352)
| method | defect | precision | recall | f1 | support | n_images |
|---|---|---|---|---|---|---|
| dip | blur | 0.872 | 0.788 | 0.828 | 52 | 352 |
| dip | noise | 0.979 | 1.000 | 0.989 | 47 | 352 |
| dip | dark | 0.804 | 0.837 | 0.820 | 49 | 352 |
| dip | non_white_bg | 0.969 | 1.000 | 0.984 | 63 | 352 |
| dip | bad_framing | 0.735 | 0.926 | 0.820 | 54 | 352 |
| dip | cropped | 0.647 | 0.702 | 0.673 | 47 | 352 |
| cnn | blur | 1.000 | 1.000 | 1.000 | 52 | 352 |
| cnn | noise | 1.000 | 1.000 | 1.000 | 47 | 352 |
| cnn | dark | 0.980 | 1.000 | 0.990 | 49 | 352 |
| cnn | non_white_bg | 1.000 | 1.000 | 1.000 | 63 | 352 |
| cnn | bad_framing | 1.000 | 1.000 | 1.000 | 54 | 352 |
| cnn | cropped | 0.979 | 0.979 | 0.979 | 47 | 352 |
| vlm | blur | 0.591 | 1.000 | 0.743 | 52 | 352 |
| vlm | noise | 0.653 | 1.000 | 0.790 | 47 | 352 |
| vlm | dark | 0.875 | 0.571 | 0.691 | 49 | 352 |
| vlm | non_white_bg | 0.516 | 1.000 | 0.681 | 63 | 352 |
| vlm | bad_framing | 0.412 | 1.000 | 0.584 | 54 | 352 |
| vlm | cropped | 0.739 | 0.362 | 0.486 | 47 | 352 |
