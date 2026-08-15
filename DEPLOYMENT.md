# RoadSense AI ML Deployment Notes

## Production Model

The current production image-classification artifact for the RoadSense AI backend is:

- Filename: `efficientnet_best_model.keras`
- Architecture: `EfficientNetB0`
- Input shape: `224 × 224 × 3`
- Preprocessing: `tensorflow.keras.applications.efficientnet.preprocess_input`
- Number of output classes: `7`
- Class order:
  1. Broken Road Sign Issues
  2. Damaged Road issues
  3. Illegal Parking Issues
  4. Littering Garbage on Public Places Issues
  5. Mixed Issues
  6. Pothole Issues
  7. Vandalism Issues
- Production model location in this repository: `models/efficientnet_best_model.keras`
- Model size: `33,574,604 bytes`
- SHA-256: `1914A24EB46CBC27B1F00BF4B61C5C04F6E36C9AD52A2905F2A2B4E09FDCA426`

This artifact has been verified to be byte-identical to the model currently consumed by the RoadSense AI backend.

## Backend Handoff

The backend does not need to retrain the model.

The backend expects to obtain this exact `.keras` artifact through a deployment artifact URL such as:

- `MODEL_DOWNLOAD_URL`

The backend may optionally verify the downloaded file using:

- `MODEL_SHA256`

The model must remain compatible with the existing backend preprocessing and the seven-class mapping described above.

## Recommended Release Process

Use a GitHub Release asset as the production handoff mechanism for the model artifact.

1. Verify that the final trained model is `models/efficientnet_best_model.keras`.
2. Calculate the SHA-256 checksum for that file.
3. Confirm the checksum matches the documented production checksum above.
4. Create a GitHub Release in the `RoadSense-AI-ML` repository.
5. Attach the model file as a release asset named `efficientnet_best_model.keras`.
6. Give the release a clear semantic version/tag such as `model-v1.0.0`.
7. Copy the direct release asset URL.
8. Configure that URL in the deployed backend as `MODEL_DOWNLOAD_URL`.
9. Configure the checksum in the deployed backend as `MODEL_SHA256`.

Do not create the GitHub Release automatically. Do not upload the model automatically. Do not commit the `.keras` model into normal Git history if it is currently intentionally ignored.

## Versioning Guidance

Create a new model version/tag only when something meaningful changes, such as:

- retraining
- architecture change
- preprocessing change
- class mapping change
- materially different dataset
- new production artifact

The current verified model can be documented as the initial production model release.

## Compatibility Warning

If future retraining changes any of the following, the backend must be updated and revalidated before deployment:

- input dimensions
- preprocessing
- number of classes
- class order
