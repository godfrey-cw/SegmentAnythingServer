# A Segment Anything Inference API running on TorchServe

## Quickstart

Recursively clone this repository with

```bash
git clone --recursive https://github.com/godfrey-cw/SegmentAnythingServer.git
```

Then check out the Bash script executed below. It was developed on an Ubuntu
20.04 VM (specifically a [Lambda Labs](https://lambdalabs.com/) A10 instance).
If your setup is different you may need to make some changes. In particular I
**do not recommend running this Bash script on your laptop/desktop** as is (it
will install a bunch of packages with `apt`, polute your system Python
environment ... not good).

Anyway, once you are comfortable run

```bash
bash SegmentAnythingServer/serve.sh
```

## In this repository

The Segment Anything models are pulled from the [official
repository](https://github.com/facebookresearch/segment-anything). In `models`
they are compartmentalized into single class files; the scripts in `handlers`
define server-side logic for different modes of operation (automatic mask
generation, prompted segmentation). Together, these model/handler files are used
to package model archive files for TorchServe.

The data preprocessing in the scripts of `handlers` has specific expectations
for how images/prompts are uploaded; see the notebook/script in `examples` for a
walkthrough of calling the API with `requests`.

## Alternatives

[developmentseed/segment-anything-services](https://github.com/developmentseed/segment-anything-services)
offers additional functionality such as separation of encoder and decoder model
components, Dockerization and Jupyter server integration. The reasons for
creating a new repository included the original author's ignorance of Docker,
not requiring separation of the encoder and decoder, and a desire to understand
and possibly tweak TorchServe handlers to explore new uses of Segment Anything.
