# Explainable Emotion Recognition from Facial Expressions

https://github.com/michaelnoi/emotion_xai/assets/41581344/22d401ef-651a-4268-bf69-4eb9ac1ef444

Source video: [Video of TikTok challenge](https://www.youtube.com/watch?v=4vZSsEs-NFk)



## Description

This project aims to develop a live emotion recognition demonstrator using GradCAM to explain the model's predictions. The model is trained on the RAF-DB dataset, which is a large-scale dataset of facial images with labeled emotions. The demonstrator uses a face detector to track faces in a live camera feed and then predicts the emotions of the faces using the trained emotion recognition model. GradCAM is used to generate a heatmap of the CNN's activation gradients, which highlights the regions of the face that the model is paying attention to when making its predictions.



## Installation

To install the project, clone the repository and install the required dependencies:

```
git clone https://github.com/michaelnoi/emotion_xai.git
cd emotion_xai
pip install -r requirements.txt
```



## Demonstrator

To run the live demonstrator, train a model and specify its path in ```live.py```. Then, run the following command:

```
python live.py
```



### Running on device

The demonstrator will run on a CPU by default. It is also able to leverage Apple Silicon GPUs via device ```"mps"``` and NVIDIA GPUs via device ```"cuda"```. 

In basic test environment on M1 chip with 8 GB shared memory, the full setup runs at 10 FPS.

To run the demonstrator on a device with a GPU, you will need to install the PyTorch build for your device. See the [PyTorch website](https://pytorch.org/get-started/locally/) for more details.



## Data

The model is trained on the RAF-DB dataset, which can be downloaded from [here](http://www.whdeng.cn/raf/model1.html) for research purposes only.

From the example videos provided, one can clearly see that happy images are overrepresented in the data giving the model a bias towards happy predictions and high confidence in making these predictions which leads to some obvious mislabeling.



## Train

To train the model, use the default RAF-DB folder structure and make sure you add the distribution of the crowd sourced votes as a ```.csv``` alongside the argmax label. The training objective is the KL divergence on the distribution.
If no label distribution is available, switch to CE loss.

Run the following command:

```
python train.py
```



## Evaluation

The evaluation notebook gives an overview of the model's performance on the test set. The test CAM notebook shows some example GradCAM heatmaps.



## License

This project is licensed under the MIT License.



## More Examples


https://github.com/michaelnoi/emotion_xai/assets/41581344/e6d6b9bd-d24d-417d-925d-be146a95fd83

https://github.com/michaelnoi/emotion_xai/assets/41581344/fd204f05-6141-4f47-a573-abcfb8a17d81

https://github.com/michaelnoi/emotion_xai/assets/41581344/3db92f31-d4bf-433d-ade0-3e59ea4d3fa2 



## References

- Dataset: Li, S., Deng, W., & Du, J. (2017). Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2852-2861).
- RetinaFace: Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). Retinaface: Single-shot multi-level face localisation in the wild. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 5203-5212).
- GradCAM: Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).

--- 

- Data from here http://www.whdeng.cn/raf/model1.html
- Face Detection setup from here https://github.com/elliottzheng/batch-face
- Model and config originally from RetinaFace https://github.com/biubug6/Pytorch_Retinaface
