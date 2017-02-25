### Python implementation of [siamese-fc](https://github.com/bertinetto/siamese-fc)

--------------

This repository only include the tracking part of [siamese-fc](https://github.com/bertinetto/siamese-fc).

--------------

#### Dependency

- Mxnet = 0.9.2
- OpenCV
- Numpy

#### Usage

Before running the demo, we should convert the `matconvnet` model to `mxnet` model. 

```
python transfer_model.py
```

By default there is already an `mxnet` model in `model` folder with prefix `mxmodel_bgr`, which means you should feed a `BGR` image to the model. If you want to use the `RGB` one, you should modify the tracking code correspondingly.

Run the default demo:

```
python run_tracker.py
```
