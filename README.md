# Flask Server
### Running YOLO

The endpoints that upload images to FastAPI are.
```[POST] /fast/upload```

When you send the image to the FastAPI server, it detects the posture of the pet due to the model we trained and shows the results.
In the result value, **name** indicates what position (ex. dog_sit) is, and **class** shows numerically how similar it is.
