# tensorflowforpascal
Use TensorFlow from Pascal (FreePascal, Lazarus, etc.)

This project is to support TensorFlow in a Pascal environment. It is built up hierarchically as follows:

Low level:

tf_api.pas (in units) is the low level pascal interface to the TensorFlow library (c_api.h alike).

Medium level:

tf_tensors.pas (in units) is a unit to create Tensors and do some basic stuff with them.

tf_operations.pas (in units) is a unit to build Graphs or to execute Operations in an Eager like way. Both the Graph and the Eager approach use operation independent, "Generic" functions to access TensorFlow operations.

High level:

tf_wrapper.pas (you can generate it for yourself, but a copy is available in units) is the highest level interface, including operation specific functions built on the tf_operations.pas Generic object TGraph and Generic function ExecOper. This unit is automatically generated using the WrapperMaker program (see below).

WrapperMaker program

wrappermaker.pas (in wrappermaker) is the source code of the experimental program WrapperMaker. It uses the TensorFlow ops.pbtxt description file (available from github.com/tensorflow, but a copy is included in wrappermaker) and the tf_wrappertemplate.pas source file (also in wrappermaker) to generate tf_wrapper. Check the source code for details or run it with --help option to see, how you can generate different versions of tf_wrapper according to your needs.

ops.pbtxt (in wrappermaker) is the copy of TensorFlow operation description file.

tf_wrappertemplate.pas (in wrappermaker) is the template used to generate tf_wrapper.

help1.txt and help2.txt (in wrappermaker) are the help files for --help=1 and --help=2 options.

Examples

examples.pas (in examples) is a simple program, showing some examples, how to use all the above.

tf_utils (in examples) is a very simple utility unit, to add useful little functions.

mnist  (in examples) is an example how to solve MNIST using the above interfaces, wrappers, etc.



Please let me know if you find it useful, if you find bugs or if you want to contribute.
