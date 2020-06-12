# 从0到1PaddlePaddle入门之
# 利用PaddleDetection训练模型
# 并用Paddle-Lite在安卓端部署
最近因为疫情原因宅在家里，就搜集了些照片用[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/ "PaddleDetection")训练了一个口罩分类的模型，摸索了一下Paddle-Lite的Andriod部署，恰好Paddle-Lite最近也有比较大的迭代更新，这篇博客记录了我的摸索过程和一点点心得。

我不太熟悉Andriod开发，此demo仅仅在Paddle-Lite-Demo[Paddle-Lite-Demo](https://github.com/PaddlePaddle/PaddleDetection/ "Paddle-Lite-Demo")的基础上替换模型，修改了少量代码，以跑通训练和部署流程为目的。


# 正文开始：

## 1.PaddlePaddle开发环境：
如果自己没有支持CUDA的GPU设备的话可以选择百度官方的[AI-Studio](https://github.com/PaddlePaddle/PaddleDetection/ "AI-Studio")。
自己有设备的话可以参照[PaddleDection安装说明](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/docs/tutorials/INSTALL_cn.md "PaddleDection安装说明")配置开发环境。
注：请注意匹配PaddlePaddle版本（1.7）和PaddleDetection分支（0.2）。

## 2.模型训练：
具体训练流程请参考PaddleDetection[官方教程](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/docs/tutorials/GETTING_STARTED_cn.md "官方教程")。
我采用的模型是[yolov3_mobilenet_v1](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/configs/yolov3_mobilenet_v1.yml "yolov3_mobilenet_v1")。如果你还没开始训练的话，请你选择yolo系列的模型，因为不同模型的输入输出有所不同。

## 3.模型导出：
假设你模型已经训练完毕并且保存在了
```bash
/_path_/_to_/_dir_/PaddleDetection/output/yolov3_mobilenet_v1_mask
```
请在PaddleDection目录下执行以下代码：
```bash
python tools/export_model.py -c configs/yolov3_mobilenet_v1_mask.yml \
        --output_dir=./inference_model \
        -o weights=output/yolov3_mobilenet_v1_mask/model_final
```
预测模型会导出到inference_model/yolov3_mobilenet_v1_mask目录下，模型名和参数名分别为__model__和__params__。
具体导出方法请参考：[这里](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/docs/advanced_tutorials/inference/EXPORT_MODEL.md "这里")。

## 4.模型转换：
### 工具准备
模型转换需要用到Paddle-Lite提供的模型转换工具：opt
这里我们使用Paddle-Lite官方发布的版本，Paddle-Lite Github仓库的[release](https://github.com/PaddlePaddle/Paddle-Lite/releases "release")界面，选择release版本下载对应的转化工具。
也可以参考[文档](https://paddle-lite.readthedocs.io/zh/latest/user_guides/model_optimize_tool.html "文档")自行编译。
### 转换模型
我们将opt工具拷贝到PaddleDetection目录下，执行以下命令：
```bash
./opt --model_file=yolov3_mobilenet_v1_mask/__model__ \
	--param_file=/yolov3_mobilenet_v1_mask/__params__ \
	--optimize_out=mask \
	--optimize_out_type=naive_buffe
```
(3) 更详尽的转化命令总结：

```bash
./opt \
    --model_dir=<model_param_dir> \
    --model_file=<model_path> \
    --param_file=<param_path> \
    --optimize_out_type=(protobuf|naive_buffer) \
    --optimize_out=<output_optimize_model_dir> \
    --valid_targets=(arm|opencl|x86|npu|xpu) \
    --prefer_int8_kernel=(true|false) \
    --record_tailoring_info =(true|false)
```

```bash
选项	说明
--model_dir	待优化的PaddlePaddle模型（非combined形式）的路径
--model_file	待优化的PaddlePaddle模型（combined形式）的网络结构文件路径。
--param_file	待优化的PaddlePaddle模型（combined形式）的权重文件路径。
--optimize_out_type	输出模型类型，目前支持两种类型：protobuf和naive_buffer，其中naive_buffer是一种更轻量级的序列化/反序列化实现。若您需要在mobile端执行模型预测，请将此选项设置为naive_buffer。默认为protobuf。
--optimize_out	优化模型的输出路径。
--valid_targets	指定模型可执行的backend，默认为arm。目前可支持x86、arm、opencl、npu、xpu，可以同时指定多个backend(以空格分隔)，Model Optimize Tool将会自动选择最佳方式。如果需要支持华为NPU（Kirin 810/990 Soc搭载的达芬奇架构NPU），应当设置为npu, arm。
--prefer_int8_kernel	若待优化模型为int8量化模型（如量化训练得到的量化模型），则设置该选项为true以使用int8内核函数进行推理加速，默认为false。
--record_tailoring_info	当使用 根据模型裁剪库文件 功能时，则设置该选项为true，以记录优化后模型含有的kernel和OP信息，默认为false。
```
> ** 如果待优化的fluid模型是非combined形式，请设置--model_dir，忽略--model_file和--param_file。
如果待优化的fluid模型是combined形式，请设置--model_file和--param_file，忽略--model_dir。
优化后的模型为以.nb名称结尾的单个文件。**

##### 到这里我们已经得到了mask.nb这个模型文件。

## 5. 准备Paddle-Lite-Demo
（1）参考[Github](https://github.com/PaddlePaddle/Paddle-Lite-Demo "Github")的readme准备demo。
在官方 [release](https://paddle-lite.readthedocs.io/zh/latest/user_guides/release_lib.html) 预编译库下载编译库并替换demo中的库，或者手动编译。
### Android更新预测库
替换jar文件：将生成的build.lite.android.xxx.gcc/inference_lite_lib.android.xxx/java/jar/PaddlePredictor.jar替换demo中的Paddle-Lite-Demo/PaddleLite-android-demo/image_classification_demo/app/libs/PaddlePredictor.jar
替换arm64-v8a jni库文件：将生成build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/java/so/libpaddle_lite_jni.so库替换demo中的Paddle-Lite-Demo/PaddleLite-android-demo/image_classification_demo/app/src/main/jniLibs/arm64-v8a/libpaddle_lite_jni.so 替换armeabi-v7a jni库文件：将生成的build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/java/so/libpaddle_lite_jni.so库替换demo中的Paddle-Lite-Demo/PaddleLite-android-demo/image_classification_demo/app/src/main/jniLibs/armeabi-v7a/libpaddle_lite_jni.so.
### 注意，一定要换的最新的预测库

（2）编译运行object_detection_demo,确保能运行。

## 6.替换原demo中的模型
原demo运行成功，接下来该换上我们自己的模型了。
#### 1. 将模型拷贝到 Paddle-Lite-Demo/PaddleLite-android-demo/object_detection_demo/app/src/main/assets/models/mask 目录下并改名为model.nb。
#### 2. 将训练模型时的mask_label_list文件拷贝到 Paddle-Lite-Demo/PaddleLite-android-demo/object_detection_demo/app/src/main/assets/labels/mask_label_list 。
#### 3. 修改 Paddle-Lite-Demo/PaddleLite-android-demo/object_detection_demo/app/src/main/res/values/strings.xml 文件。
    <string name="MODEL_PATH_DEFAULT">models/mask</string>
    <string name="LABEL_PATH_DEFAULT">labels/mask_label_list</string>
    
    <string name="INPUT_SHAPE_DEFAULT">1,3,320,320</string>
    <string name="INPUT_MEAN_DEFAULT">0.485,0.456,0.406</string>
    <string name="INPUT_STD_DEFAULT">0.229,0.224,0.225</string>

#### 4.修改代码：
将：
```java
protected long[] inputShape = new long[]{1, 3, 300, 300};
protected float[] inputMean = new float[]{0.5f, 0.5f, 0.5f};
protected float[] inputStd = new float[]{0.5f, 0.5f, 0.5f};
```
修改为：
```java
protected long[] inputShape = new long[]{1, 3, 320, 320};
protected float[] inputMean = new float[]{0.485f, 0.456f, 0.406f};
protected float[] inputStd = new float[]{0.229f, 0.224f, 0.225f};
```
** *其中320为模型中图片输入的大小，如果你的模型为608，请改成608。**

------------
修改模型输入部分为：
```java
        // Set input shape
        Tensor inputTensor0 = getInput(0);
        inputTensor0.resize(inputShape);
        Tensor inputTensor1 = getInput(1);
        inputTensor1.resize(new long[] {1,2});
        // Pre-process image, and feed input tensor with pre-processed data
        Date start = new Date();
        int channels = (int) inputShape[1];
        int width = (int) inputShape[3];
        int height = (int) inputShape[2];
        float[] inputData = new float[channels * width * height];
        if (channels == 3) {
            int[] channelIdx = null;
            if (inputColorFormat.equalsIgnoreCase("RGB")) {
                channelIdx = new int[]{0, 1, 2};
            } else if (inputColorFormat.equalsIgnoreCase("BGR")) {
                channelIdx = new int[]{2, 1, 0};
            } else {
                Log.i(TAG, "Unknown color format " + inputColorFormat + ", only RGB and BGR color format is " +
                        "supported!");
                return false;
            }
            int[] channelStride = new int[]{width * height, width * height * 2};
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int color = inputImage.getPixel(x, y);
                    float[] rgb = new float[]{(float) red(color) / 255.0f, (float) green(color) / 255.0f,
                            (float) blue(color) / 255.0f};
                    inputData[y * width + x] = (rgb[channelIdx[0]] - inputMean[0]) / inputStd[0];
                    inputData[y * width + x + channelStride[0]] = (rgb[channelIdx[1]] - inputMean[1]) / inputStd[1];
                    inputData[y * width + x + channelStride[1]] = (rgb[channelIdx[2]] - inputMean[2]) / inputStd[2];
                }
            }
        } else if (channels == 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int color = inputImage.getPixel(x, y);
                    float gray = (float) (red(color) + green(color) + blue(color)) / 3.0f / 255.0f;
                    inputData[y * width + x] = (gray - inputMean[0]) / inputStd[0];
                }
            }
        } else {
            Log.i(TAG, "Unsupported channel size " + Integer.toString(channels) + ",  only channel 1 and 3 is " +
                    "supported!");
            return false;
        }
        inputTensor0.setData(inputData);
        inputTensor1.setData(new int[] {320,320});
        Date end = new Date();
        preprocessTime = (float) (end.getTime() - start.getTime());
```
修改模型的输出的处理部分：
*yolo 的模型输出为坐标值，ssd为坐标点的相对值，这里统一/320 就转换成了相对值，省去了调整加框部分代码的麻烦。*
```java
        // Post-process
        start = new Date();
        long outputShape[] = outputTensor.shape();
         long outputSize = 1;
        for (long s : outputShape) {
            outputSize *= s;
        }
        outputImage = inputImage;
        outputResult = new String();
        Canvas canvas = new Canvas(outputImage);
        Paint rectPaint = new Paint();
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(1);
        Paint txtPaint = new Paint();
        txtPaint.setTextSize(12);
        txtPaint.setAntiAlias(true);
        int txtXOffset = 4;
        int txtYOffset = (int) (Math.ceil(-txtPaint.getFontMetrics().ascent));
        int imgWidth = outputImage.getWidth();
        int imgHeight = outputImage.getHeight();
        int objectIdx = 0;
        final int[] objectColor = {0xFFFF00CC, 0xFFFF0000, 0xFFFFFF33, 0xFF0000FF, 0xFF00FF00,
                0xFF000000, 0xFF339933};
        for (int i = 0; i < outputSize; i += 6) {
            float score = outputTensor.getFloatData()[i + 1];
            if (score < scoreThreshold) {
                continue;
            }
            int categoryIdx = (int) outputTensor.getFloatData()[i];
            String categoryName = "Unknown";
            if (wordLabels.size() > 0 && categoryIdx >= 0 && categoryIdx < wordLabels.size()) {
                categoryName = wordLabels.get(categoryIdx);
            }
            float rawLeft = outputTensor.getFloatData()[i + 2]/320;
            float rawTop = outputTensor.getFloatData()[i + 3]/320;
            float rawRight = outputTensor.getFloatData()[i + 4]/320;
            float rawBottom = outputTensor.getFloatData()[i + 5]/320;
            float clampedLeft = Math.max(Math.min(rawLeft, 1.f), 0.f);
            float clampedTop = Math.max(Math.min(rawTop, 1.f), 0.f);
            float clampedRight = Math.max(Math.min(rawRight, 1.f), 0.f);
            float clampedBottom = Math.max(Math.min(rawBottom, 1.f), 0.f);
            float imgLeft = clampedLeft * imgWidth;
            float imgTop = clampedTop * imgWidth;
            float imgRight = clampedRight * imgHeight;
            float imgBottom = clampedBottom * imgHeight;
            int color = objectColor[objectIdx % objectColor.length];
            rectPaint.setColor(color);
            txtPaint.setColor(color);
            canvas.drawRect(imgLeft, imgTop, imgRight, imgBottom, rectPaint);
            canvas.drawText(objectIdx + "." + categoryName + ":" + String.format("%.3f", score),
                    imgLeft + txtXOffset, imgTop + txtYOffset, txtPaint);
            outputResult += objectIdx + "." + categoryName + " - " + String.format("%.3f", score) +
                    " [" + String.format("%.3f", rawLeft) + "," + String.format("%.3f", rawTop) + "," + String.format("%.3f", rawRight) + "," + String.format("%.3f", rawBottom) + "]\n";
            objectIdx++;
        }
        end = new Date();
        postprocessTime = (float) (end.getTime() - start.getTime());
        return true;

```
至此，Andriod端的部署就完成了。试着运行一下吧！

## 附
关于Andriod端部署：
直接替换模型不修改代码的话是跑不通的，主要是因为属于没有预处理成模型的能接收的数据。
这里表现在:
```
原本SSD模型的输入为：
	im [1,3,300,300]
而yolo的模型输入要求为：
	input0: im [1,3,320,320]
	input1: im_sz[320,320]
```
在替换模型之后记得要修改模型的与处理部分，以及模型输出的处理部分。

如果对模型的部署还有问题，欢迎大家来paddle-lite官方群（696965088）和小伙伴们一起探讨。

apk下载链接:https://pan.baidu.com/s/1uWTRb0EvV6gQJF8x8D2pPQ  密码:utl2

注：仅供测试使用，非百度官方发布模型
