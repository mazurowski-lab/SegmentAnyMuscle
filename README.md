# Segment Anything Muscle

#### By Roy Colglazier*, Jisoo Lee*, [Haoyu Dong*](https://haoyudong-97.github.io/), Hanxue Gu, Yaqian Chen, Zafer Yildiz, Zhonghao Liu, Nicholas Konz, Jichen Yang, Jikai Zhang, Yuwen Chen, Lin Li, Haoran Zhang, Joseph Cao, Adrian Camarena, Maciej Mazurowski.

## Guideline for Implement Fine-tuned SAM

### Get the model weights
Weights for fine-tuned SAM can be accessed [here](https://drive.google.com/file/d/1mpTW0TgLgkRIG3sdx9ys5r2iW6lAJw5u/view?usp=sharing). It should be placed under checkpoints/ folder.

### Run the evaluation code
First, you need to define your input and output path in Line 27 and 28. The input folder is expected to contain volume inputs, i.e., XXX.nii.gz. 
Then, you can simply run the evaluation with
```
python3 evaluate.py
```
The output will be saved in numpy format (.npy) and nearly raw raster data format (.seg.nrrd).

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
