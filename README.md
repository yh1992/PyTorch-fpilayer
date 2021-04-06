# PyTorch-fpilayer

Copyright (C) 2021 Younghan Jeon, Minsik Lee

This code is an implementation of the methods described in:

    Younghan Jeon, Minsik Lee, and Jin Young Choi,
    "Differentiable Forward and Backward Fixed-point Iteration Layers",
    IEEE Access, January 22, 2021.

This software is distributed WITHOUT ANY WARRANTY. Use of this software is 
granted for research conducted at research institutions only. Commercial use
of this software is not allowed. Corporations interested in the use of this
software should contact the authors. If you use this code for a scientific
publication, please cite the above paper.

Dataset files "bibtex_train_final.csv" and "bibtex_train_final.csv" are converted versions of the original
arff files from http://mulan.sourceforge.net/datasets-mlc.html which is a multi-label text
dataset described in:

    I. Katakis, G. Tsoumakas, I. Vlahavas,
    "Multilabel Text Classification for Automated Tag Suggestion",
    Proceedings of the ECML/PKDD 2008 Discovery Challenge, Antwerp, Belgium, 2008.

Each file is '(# of data) by 1995 zero-one matrix' and must be placed in the "data" folder.

USAGE:

Please see the demo files "FPI_GD" and "FPI_NN" for usage information. These scripts were
tested with python 3.6.

FEEDBACK:

Your feedback is greatly welcome. Please send bug reports, suggestions, and/or
new results to:

    yh1992@snu.ac.kr

CONTENTS:

    README.md:                         This file.
    LICENSE.md:                        License information.
    fixed_point_iteration.py:          Base algorithm of our method.
    FPI_NN.py:                         Demo with FPI_NN for multi-label classification.
    FPI_GD.py:                         Demo with FPI_GD for multi-label classification.
    model/pretrained_nn_0.434:         Pretrained model of FPI_NN.
    model/pretrained_gd_0.432:         Pretrained model of FPI_GD.
