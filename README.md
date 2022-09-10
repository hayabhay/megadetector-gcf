# Cloud function to run MegaDetector

This function runs MegaDetector inference on a single image.
Image source can be either a web url, a google storage uri (needs access to the project)
or raw binary image data.

Objects, if detected, can be found in the "detections" key in the returned payload.

For local dev, run the following command inside the `megadetector` directory after installing`requirements-dev.txt`

```
pip install -r requirements-dev.txt
functions-framework --target detect --debug
```

Be sure to run pre-commit before committing the code
```
git add -u
pre-commit run --all
```
Note: This is mostly for formatting and other glaring gotchas. Feel free to ignore overly specific style issues.

Deploy using
```
gcloud functions deploy <function_name> --entry-point detect --runtime python39 --trigger-http --region=us-west2 --memory=8192 --timeout=540s
```
