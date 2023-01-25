The wrappermaker folder is already updated to use the latest (as of 18/01/2023) ops.pbtxt file and generate the interface for TF 2.11.
Also the tf_wrapper.pas (in units) is the latest generated.
There is a significant change compared to pre 25/01/2023 versions: Now Operations with mixed Inputs and InputLists are handled correctly.

All other units in the units folder is up-to-date.
In tf_operations both the TGraph.AddOper and ExecOper got many new overloaded versions while the original long versions were deprecated (but still available for compatibility reasons) as those assumed that Inputs are always before InputLists and Outputs are always before OutputLists what turned out not to be the case.
Be aware that because now Inputs and InputLsits are mixed up and it would be very low performance to allow different types when passing the new InputsAndInputLists, all Inputs are passed as InputLists with just one element and a separate array can or must (depending on the overloaded version) be given, indicating which one is an Input and which one is an InputList.

In the exampes folder the source files (tf_utils.pas, examples.pas and mnist.pas) are updated and two files to test are also added. The input files for mnist has to be downloaded separately (because of their size I do not want to add them here unnecessarily).
