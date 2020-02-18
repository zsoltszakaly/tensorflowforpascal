unit tf_wrapper;

//**********************************************************************************************************************************
//
//  Pascal methods to create and manage TensorFlow based Operations in a Specific manner
//
//  Copyright: (C) 2020, Zsolt Szakaly
//
//  This source is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as
//  published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
//
//  This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
//  A copy of the GNU General Public License is available on the World Wide Web at <http://www.gnu.org/copyleft/gpl.html>. You can
//  also obtain it by writing to the Free Software Foundation, Inc., 51 Franklin Street - Fifth Floor, Boston, MA 02110-1335, USA.
//
//  Change log:
//    11/02/2020 Initial version
//    15/02/2020 ctypes add to uses
//
//**********************************************************************************************************************************
//
//  Description
//  #TemplateDescriptionBegin
//
//  When you see this text, it means that you look into the tf_wrappertemplate.pas As the name suggests, it is only a template file
//  to be used by the WrapperMaker program, that can generate (depending on its CommandLine parameters - see its Help for details)
//  operation specific interfaces both for the TensorFlow Graph approach and the Eager approach.
//  This file has labels starting with #. Please do not remove them, or the WrapperMaker will not be able to use it.
//  The rest of the description and comments below are for the generated tf_wrapper unit, so read it accordingly.
//
//  #TemplateDescriptionEnd
//  #GraphDescriptionBegin
//
//  This wrapper file includes a TGraphExt object, built on the TGraph object defined in tf_operations. This extension of the
//  TGraph adds operation specific Add<oper> methods. Using them is functionally the same as using the original AddOper function.
//  There are some benefits though. Already at the time typing the Add<oper>( the editor (if you use an IDE, like Lazarus) will
//  bring up the parameter list, so you can see what inputs and parameters are needed for a given operation. This can simplify
//  the programming work. On the other hand, it must be mentioned that certain TensorFlow operations have default attributes. E.g.
//  MatMul (matrix multiplication) have two attributes (transpose_a and transpose_b) indicating whether the input matrix/matrices
//  need to be transposed before the operation or not. The default value for both, is false. If you use AddOper, you can safely skip
//  these attributes if you do not want to transpose your matrices. In the Specific calls (e.g. AddMatMul), included in this file
//  all attributes of the given operation must be specified. There were considerations to avoid this, but none of the options is
//  practical to implement. The "elegant" way of having "transpose:boolean=false" instead of just having "transpose:boolean" in the
//  declaration could be done. The problem with this approach is that there are operations that use a Tensor as an attribute and it
//  even has a default, but it is not possible to use something like "const Attr:TF_Tensor=CreateTensor". For the majority of
//  attributes with default however this approach could work, but in this case the default value would already be in tf_wrapper and
//  if TensorFlow changes it in the library, the wrapper would still call it with the old default. This was not a desired outcome.
//  Another approach could be to have different versions of the definition. In case of the above mentioned MatMul, there could be
//  three different versions, one with no transpose atribute, one with one transpose atribute and one with two transpose attributes.
//  The problem that in case of one, it can only be either the first or the second, but there is no way do define versions depending
//  which one you want to give. For more complicated operations with more attributes the number of combinations increases rapidly,
//  especially when the attributes have different types. So, again this was not a desired outcome. The third way is even uglier.
//  There could be functions with different names, depending which attribute is specified and which is not. This would make the use
//  probably even more complicated, e.g. having AddMatMul, AddMatMulTransposea, AddMatMulTransposeb, AddMatMulTransposeaTransposeb.
//  Oddly enough in this case the name could already indicate even the value of the attribute (using the one with Transposea
//  probably means that you want to transpose a, so it is not needed as an attribute. For more comlex operations the number of such
//  name combinations would be 2^n, what is again not a desired outcome. This is why the Add<oper> functions include all the
//  attributes, regardless if they have default value or not.
//  It is possible, that later the WrapperMaker will be improved in a way, that at least for the easy cases some simplification is
//  added, but it is not a priority. If for whatever reason you cannot specify the attribute and you want to use the TensorFlow
//  default value, you can still use the base AddOper function. AddOper and the different Add<oper> functions can be used together,
//  so if only for one operation you have this problem with, you can still use TGraphExt and use the different Add<oper> functions.
//
//  #GraphDescriptionEnd
//  #EagerDescriptionBegin
//
//  This wrapper file includes operation specific Exec<oper> functions. These functions are built on the Generic ExecOper function
//  defined in tf_operations. Exec<oper> functions are only generated for operations that have only ONE Output and no InputList or
//  OutputList parameters. In the background the Specific Exec<oper> functions call the short version of the  ExecOper function.
//  The benefit of using Exec<oper> is that it lists what parameters are required to call the given operation, let it be Inputs or
//  Attributes. If you use an IDE, already at typing Exec<oper>(, you can see the parameter list, do not need to look it up in the
//  specification. Also, if an Attribute can be retrieved from the Input Tensor (typically the data type of it, often called "T"),
//  then it is automatically done, you do not need to input it.
//  As described in the Description of the TGraphExt.Add<oper> (if this tf_wrapper was generated without generating TGraphExt, then
//  you can find this description in the tf_wrappertemplate file) there is a difference between how ExecOper and Exec<oper> handle
//  the TensorFlow default attributes. In ExecOper you can skip those attributes that have a default value and you are happy with
//  it. In the Exec<oper> functions you have to explicitely give all attributes of the given operations. The reasons can be found in
//  the mentioned TGraphExt description. Similarly to TGraphExt this is currently a low priority to improve on this. It is not a bad
//  programming practice anyway to specify these parameters explicitely, but if you want to avoid it, just use the base ExecOper.
//  Just like, in case of ExecOper, all Exec<oper> functions have optional boolean parameters (starting with D_) to delete the input
//  Tensors at the end of the Operation. This is to make memory management much easier and allow in-line TensorCreate when calling
//  ExecOper or one of the Exec<oper> functions, without creating a memory leak.
//
//  #EagerDescriptionEnd
//**********************************************************************************************************************************


interface

uses
  ctypes,
  SysUtils,
  tf_api,
  tf_operations;

//  #GraphInterfaceBegin

// The new TGraphExt object based on the TGraph, specified in tf_operations
type
  TGraphExt=object(TGraph)
//  #GraphInterfaceFill
    end;
//  #GraphInterfaceEnd

//  #EagerInterfaceBegin

// The new Exec<oper> functions, built on top of the short version of ExecOper function, specified in tf_operations

//  #EagerInterfaceFill

//  #EagerInterfaceEnd

implementation

//  #GraphImplementationBegin
//  The TGraphExt methods
//  #GraphImplementationFill
//  #GraphImplementationEnd

//  #EagerImplementationBegin
//  The Exec<oper> methods
//  #EagerImplementationFill
//  #EagerImplementationEnd

end.

