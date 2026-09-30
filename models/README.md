# Model files (v14.6)

| File | Use |
|---|---|
| final_model_v14.6_clean_allyears.pkl | Live model (trained on all years) |
| final_model_v14.6_clean.pkl | Trained through 2024. Gave the reported test scores. |
| final_model_v14.6_clean_info.json | Feature list and alert threshold (0.770 on the raw score) |
| final_model_v14.6_calibrator.json | Raw score -> estimated chance (plain numbers) |

The .pkl files were saved with lightgbm 4.6.0, scikit-learn 1.6.1, python 3.13.15.
Load them with the same (or close) versions. Pickle files only load from sources you trust; these are your own.
See docs/model_card.md for how to use them.
