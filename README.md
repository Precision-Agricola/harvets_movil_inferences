# Object detection app using YOLOv8 and Android

## Step 1 (Train and export Object detection model):
- git clone https://github.com/AarohiSingla/Object-Detection-Android-App.git
  
- Train yolov8 model on custom dataset and export it in .tflite format. (Check train_export_yolov8_model.ipynb )

## Step 2 (Object detection android app setup):
- Open android_app folder.

- Put your .tflite model and .txt label file inside the assets folder. You can find assets folder at this location: <b> android_app\android_app\app\src\main\assets</b>

- Rename paths of your model and labels file in Constants.kt file. You can find Constants.kt at this location: <b>android_app\android_app\app\src\main\java\com\surendramaran\yolov8tflite </b>

- Download and install Android Studio from the official website (https://developer.android.com/studio)

- Once installed, open Android Studio from your applications menu.

- When Android Studio opens, you'll see a welcome screen. Here, you'll find options to create a new project, open an existing project, or check out project from version control.Since you already have a project, click on "Open an existing Android Studio project".

- Navigate to the directory where your project is located and select the project's root folder. 

- Build and Run
![b](https://github.com/AarohiSingla/Object-Detection-Android-App/assets/60029146/bca02a66-3421-42d9-a16f-80d5448c920f)



## Credits

This project includes code from the following repository:
