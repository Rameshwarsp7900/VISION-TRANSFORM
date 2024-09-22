% Options for packages loaded elsewhere
\PassOptionsToPackage{unicode}{hyperref}
\PassOptionsToPackage{hyphens}{url}
%
\documentclass[
]{article}
\usepackage{amsmath,amssymb}
\usepackage{lmodern}
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
  \usepackage{textcomp} % provide euro and other symbols
\else % if luatex or xetex
  \usepackage{unicode-math}
  \defaultfontfeatures{Scale=MatchLowercase}
  \defaultfontfeatures[\rmfamily]{Ligatures=TeX,Scale=1}
\fi
% Use upquote if available, for straight quotes in verbatim environments
\IfFileExists{upquote.sty}{\usepackage{upquote}}{}
\IfFileExists{microtype.sty}{% use microtype if available
  \usepackage[]{microtype}
  \UseMicrotypeSet[protrusion]{basicmath} % disable protrusion for tt fonts
}{}
\makeatletter
\@ifundefined{KOMAClassName}{% if non-KOMA class
  \IfFileExists{parskip.sty}{%
    \usepackage{parskip}
  }{% else
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{6pt plus 2pt minus 1pt}}
}{% if KOMA class
  \KOMAoptions{parskip=half}}
\makeatother
\usepackage{xcolor}
\usepackage{longtable,booktabs,array}
\usepackage{calc} % for calculating minipage widths
% Correct order of tables after \paragraph or \subparagraph
\usepackage{etoolbox}
\makeatletter
\patchcmd\longtable{\par}{\if@noskipsec\mbox{}\fi\par}{}{}
\makeatother
% Allow footnotes in longtable head/foot
\IfFileExists{footnotehyper.sty}{\usepackage{footnotehyper}}{\usepackage{footnote}}
\makesavenoteenv{longtable}
\usepackage{graphicx}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
\makeatother
% Scale images if necessary, so that they will not overflow the page
% margins by default, and it is still possible to overwrite the defaults
% using explicit options in \includegraphics[width, height, ...]{}
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
% Set default figure placement to htbp
\makeatletter
\def\fps@figure{htbp}
\makeatother
\usepackage[normalem]{ulem}
\setlength{\emergencystretch}{3em} % prevent overfull lines
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setcounter{secnumdepth}{-\maxdimen} % remove section numbering
\ifLuaTeX
  \usepackage{selnolig}  % disable illegal ligatures
\fi
\IfFileExists{bookmark.sty}{\usepackage{bookmark}}{\usepackage{hyperref}}
\IfFileExists{xurl.sty}{\usepackage{xurl}}{} % add URL line breaks if available
\urlstyle{same} % disable monospaced font for URLs
\hypersetup{
  hidelinks,
  pdfcreator={LaTeX via pandoc}}

\author{}
\date{}

\begin{document}

\begin{quote}
\textbf{VISION TRANSFORM}\\
BY THE CIFAR 10 DATASET
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image1.png}

\includegraphics[width=6.5in,height=2in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image2.png}

\begin{quote}
\textbf{ABSTRACT}

This project implements a Vision Transformer (ViT) for image
classification using the CIFAR-10 dataset. This is a type of deep
learning model that has shown good results in computer vision tasks.

The code starts by installing the necessary libraries like TensorFlow,
Keras and TensorFlow Addons. Then, it loads the CIFAR-10 dataset,
defines the model parameters and performs data augmentation. The core of
the project is building the ViT model, which involves splitting images
into patches, encoding them, applying transformer layers and finally
classifying the image.
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image3.png}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
\textbf{OBJECTIVE}

● Implement a Vision Transformer (ViT) model for image classification.
This involves understanding the architecture and building the model
using libraries like TensorFlow and Keras.

● Train and evaluate the ViT model on the CIFAR-10 dataset. This
includes data preprocessing, augmentation, training, and assessing the
model\textquotesingle s\\
performance using metrics like accuracy.

● Explore the effectiveness of ViT for image classification on a smaller
dataset. The CIFAR-10 dataset, with its limited data, allows us to study
how ViT performs in such scenarios.

● Gain practical experience in implementing a relatively new deep
learning architecture. ViT is a recent advancement in the field, and
this project offers hands-on experience with this architecture.

\textbf{Introduction}

This project implements a Vision Transformer (ViT) for image
classification using the CIFAR-10 dataset. This is a type of deep
learning model that has shown good results in computer vision tasks.

The code starts by installing the necessary libraries like TensorFlow,
Keras and TensorFlow Addons. Then, it loads the CIFAR-10 dataset,
defines the model parameters and performs data augmentation. The core of
the project is building the ViT model, which involves splitting images
into patches, encoding them, applying transformer layers and finally
classifying the image.
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
2
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
\textbf{CIFAR 10}

The CIFAR-10 dataset is a collection of small images commonly used to
train machine learning and computer vision algorithms.
Here\textquotesingle s a summary of its key characteristics:
\end{quote}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
●\\
●\\
●\\
●\\
●\\
●
\end{quote}\strut
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
\begin{quote}
Number of Images: 60,000\\
Image Size: 32x32 pixels\\
Color: Color images\\
Number of Classes: 10\\
Classes: Airplane, automobile, bird, cat, deer, dog, frog, horse, ship,
truck Training/Test Split: 50,000 training images and 10,000 test images
\end{quote}\strut
\end{minipage} \\
\midrule()
\endhead
\bottomrule()
\end{longtable}

\begin{quote}
The CIFAR-10 dataset is widely used for its manageable size and diverse
set of classes, making it suitable for experimenting with different deep
learning models and techniques.

\textbf{METHODOLOGY}

\textbf{1. Environment Setup:}

Install necessary libraries like TensorFlow, Keras, and TensorFlow
Addons.

\textbf{2. Dataset Preparation:}

a. Load the CIFAR-10 dataset using
``\uline{keras.datasets.cifar10.load\_data()}''.

b. Normalize pixel values to be in the range {[}0, 1{]}.\\
\textbf{3. Data Augmentation:}

a. Use keras.Sequential to define a data augmentation pipeline.

b. Include random flipping, cropping, and color jittering.
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
3
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
\textbf{4. Vision Transformer (ViT) Model Implementation:}

\textbf{●} \textbf{Patching:}

Divide input images into smaller patches.

\textbf{●} \textbf{Patch Encoding:}

Flatten patches and project them to a higher dimensional space. Add

positional embeddings.

\textbf{●} \textbf{Transformer Encoder:}

Utilize multiple transformer layers with multi-head attention and MLP
blocks.

\textbf{●} \textbf{Classification Head:}

Add an MLP with a final classification layer.

\textbf{5. Model Training:}

a. Define an optimizer (e.g., AdamW) and learning rate schedule.

b. Compile the model with an appropriate loss function (e.g., sparse
categorical

cross-entropy).

c. Train the model on the training data, potentially using techniques
like early

stopping and learning rate decay.

\textbf{6. Model Evaluation:}

a. Evaluate the trained model on the CIFAR-10 test set.

b. Report metrics such as accuracy, top-5 accuracy, precision, recall,
and

F1-score.
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
4
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
\textbf{Algorithm}

\textbf{1. Split Image into Patches:}\\
Divide the input image into fixed-size patches.

\textbf{2. Patch Embedding:}\\
a. Flatten each patch into a 1D vector.

b. Apply a linear transformation to map the vector to a
higher-dimensional embedding space.

c. Add positional embeddings to encode the location of each patch within
the original image.

\textbf{3. Transformer Encoder:}\\
\textbf{a. Layer Normalization:}\\
Normalize the input embeddings.

\textbf{b. Multi-Head Self-Attention:}\\
Calculate attention weights to capture relationships between different
patches.

\textbf{c. Add \& Norm:}\\
Add the attention output to the input embeddings and perform layer
normalization.

\textbf{d. Feed Forward Network:}\\
Apply a fully connected network to each embedding.
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
5
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
\textbf{e. Add \& Norm:}

Add the feed forward output to the previous output and perform layer

normalization.

\textbf{4. Transformer Encoder:}

a. Extract the embedding corresponding to a special classification token
(added

during embedding).

b. Pass this embedding through a Multilayer Perceptron (MLP) head.

c. Apply a softmax function to obtain class probabilities.

\textbf{CODE}
\end{quote}

\includegraphics[width=6.5in,height=0.82222in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image5.png}

\includegraphics[width=6.5in,height=1.08333in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image6.png}

\includegraphics[width=6.5in,height=1.38472in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image7.png}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
6
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\includegraphics[width=6.5in,height=1.04167in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image8.png}

\includegraphics[width=6.5in,height=2.46944in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image9.png}

\includegraphics[width=6.5in,height=2.5in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image10.png}

\includegraphics[width=6.5in,height=0.94722in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image11.png}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
7
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\includegraphics[width=6.5in,height=2.54167in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image12.png}

\includegraphics[width=6.5in,height=4.19722in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image13.png}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
8
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\includegraphics[width=6.5in,height=2.21805in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image14.png}

\includegraphics[width=6.5in,height=5.00972in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image15.png}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
9
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\includegraphics[width=6.5in,height=5.95833in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image16.png}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
10
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\includegraphics[width=6.5in,height=5.65556in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image17.png}

\includegraphics[width=6.5in,height=0.49028in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image18.png}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
11
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
\textbf{OUTPUT:}
\end{quote}

\begin{longtable}[]{@{}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}
  >{\raggedright\arraybackslash}p{(\columnwidth - 2\tabcolsep) * \real{0.5000}}@{}}
\toprule()
\begin{minipage}[b]{\linewidth}\raggedright
\includegraphics[width=1.88611in,height=2.16667in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image19.png}
\end{minipage} & \begin{minipage}[b]{\linewidth}\raggedright
\includegraphics[width=1.76944in,height=1.80139in]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image20.png}
\end{minipage} \\
\midrule()
\endhead
\bottomrule()
\end{longtable}

\textbf{With the accuracy of 89.37\%}

\begin{quote}
\textbf{CONCLUSION:}

This project successfully implemented a Vision Transformer (ViT) model
for image classification using the CIFAR-10 dataset. The model was
trained and evaluated, demonstrating the potential of ViT architectures
for image recognition tasks. You should execute the code to see the
final accuracy of the model.

While the achieved accuracy might be lower than state-of-the-art CNNs on
CIFAR-10, ViT offers several advantages, such as its ability to capture
global context and its scalability to larger datasets and models.
Further experimentation with hyperparameter tuning, architecture
modifications, and training strategies could lead to improved
performance.

This project provides valuable insights into the application of ViT
models for image classification and highlights their potential as a
powerful alternative to traditional CNN-based approaches.
\end{quote}

\includegraphics[width=6.5in,height=\textheight]{vertopal_99ac2e9fb6d34104a109b95b99dc0cf4/media/image4.png}

\begin{quote}
12
\end{quote}

\end{document}
