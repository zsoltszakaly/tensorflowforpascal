program examples;

//**********************************************************************************************************************************
//
//  Pascal example to use TensorFlow through Graph and Eager methods, both Generic and operation Specific
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
//    13/02/2020 Initial version
//    17/01/2023 New examples upto Example 25
//               All examples reviewed and updated if needed, consider the file as a new one (hence earlier Change logs removed)
//
//**********************************************************************************************************************************
//
//  Description
//
//  There are different examples in this file for simple matrix operations.
//  Some used the TensorFlow "native" Graph based approach where first a Graph has to be built and then that Graph can be executed
//  through a Session. To add Nodes to the Graph there are two ways. Either use the TGraph.AddOper method, that can cater -
//  hopefully - for all possible Operation types, OR use the specific TGraphExt.Add<oper> methods that are somewhat easier. Need to
//  be noted though, that while for the Generic AddOper you can easily leave out Attributes with Default values, for the specific
//  Add<oper> methods you have to add all attributes, regardless if it has default or no. Later the WrapperMaker program, that
//  generates the tf_wrapper unit, can be improved to have different versions of Add<oper>, but it is not a priority.
//  Some other examples use the Eager approach, where operations are executed immediately. Here again, you have two options; either
//  use the Generic ExecOper method or the specific Exec<oper> methods. The ExecOper can be found in tf_operations in two versions,
//  one is a full version, while the second is somewhat simplified for those operations that have no InputList input and that has
//  only one Output. The Specific Exec<oper> methods can be found in tf_wrapper and are generated automatically by WrapperMaker. As
//  in the case of the Graph operations, here again, for the Generic ExecOper you can skip default values, while for the specific
//  Exec<oper> methods you have to give all parameters. This again can be improved later. Also some of the parameters of Exec<oper>
//  can be automatically calculated, e.g. "T" is typically the TensorType of the first input Tensor. These attributes are
//  automatically generated by the relevant Exec<oper> and so, you do not need to give it. Both ExecOper and Exec<oper> have
//  additional and optional parameters to delete the input Tensors once the execution is finished. This is a big help to simplify
//  your program (avoid memory leak). Finally it must be mentioned that Exec<oper> type methods are only generated for operations
//  with exactly ONE Output and NO InputList or OutputList parameters. If you want to use such an operation in Eager mode, you must
//  use the longer version of the Generic ExecOper function.
//
//**********************************************************************************************************************************

uses
  sysutils,
  tf_api,              // the basic interface to tensorflow, based on c_api.h
  tf_tensors,          // the unit to manipulate tensors (TF_TensorPtr)
  tf_operations,       // the unit to handle Graphs, Operations and Sessions in a TensorFlow style (use Oper names)
  tf_wrapper,          // the unit where all Operations are explicitely interfaced for Graph and/or Eager use
  tf_utils;            // some very basic printing routines

procedure Example1;
// In this example a simple matrix multiplication is done on two Int32 Tensors, using the Graph based General interface
  var
    t1:TF_TensorPtr;   // The two input tensors
    t2:TF_TensorPtr;
    g:TGraph;          // The TGraph object we will use
    s:TSession;        // The TSession object we use to run
    attr:TF_DataType;  // For the Generic AddOper all Attrubute Values are handed over as pointer and so a variable is needed
    tout:TF_TensorPtr; // The result of TSession.Run
  begin
  writeln('Starting Example 1');
  writeln;
  t1:=CreateTensorInt32([3,2],[1,2,3,4,5,6]);     // A 3x2 matrix, and the data is filled through a vector
  t2:=CreateTensorInt32([2,4],[1,2,3,4,5,6,7,8]); // A 2x4 matrix, filled the same way
  g.Init;                                         // Need to call Init before the first use
  attr:=TF_TensorType(t1);                        // Set the Attribute Value for Attribute "T"
  if g.AddOper('Placeholder',[],[],['tensor1'],['dtype'],['type'],[@attr])='' then // The Generic call to add and Input
    writeln('Error while adding tensor1');
  if g.AddInput('tensor2',TF_Int32)='' then      // The same as tensor1, but with a simplified function created to add Inputs
    writeln('Error while adding tensor2');
  if g.AddOper('MatMul',['tensor1','tensor2'],[],['tensorout'],['T'],['type'],[@attr])='' then // The Generic call
    writeln('Something went wrong!');
  s.Init(g);                                      // Need to make a Session that runs the Graph
  tout:=s.run(['tensor1','tensor2'],[t1,t2],'tensorout'); // The actual run of the Session
  PrintTensorShape(tout,'tOut');           // A quick control that the product matrix is indeed 3x4, Int32
  PrintTensorData(tout);
  s.Done;                                         // Release the Session and in it the SessionPtr
  g.Done;                                         // Release the Graph and in it the GraphPtr, the StatusPtr and the Outputs
  TF_DeleteTensor(t1);                            // need to delete all the three tensors to avoid memory leak
  TF_DeleteTensor(t2);
  TF_DeleteTensor(tout);
  writeln('Finished Example 1');
  writeln;
  end;

procedure Example2;
// In this example a simple matrix multiplication is done on two Single Tensors, one of them is Transposed before multiplication
// It is done using the long version of the Eager type General interface (ExecOper)
  var
    t1:TF_TensorPtr;
    t2:TF_TensorPtr;
    attr1:TF_DataType; // For the Generic ExecOper all Attrubute Values are handed over as pointer as well
    attr2:boolean;
    tout:TF_TensorPtr; // The result of TSession.Run
  begin
  writeln('Starting Example 2');
  writeln;
  t1:=CreateTensorSingle([2,3],[1.0,2.0,3.0,4.0,5.0,6.0]);     // A 2x3 matrix, need to transpose before MatMul
  t2:=CreateTensorSingle([2,4],[1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8]); // A 2x4 matrix,no need to transpose
  attr1:=TF_TensorType(t1);
  attr2:=true;
  tout:=ExecOper('MatMul',[t1,t2],['T','transpose_a'],['type','bool'],[@attr1,@attr2],[false,false]);
  PrintTensorShape(tout,'TOut');                 // A quick control that the product matrix is indeed 3x4, Int32
  PrintTensorData(tout);
  TF_DeleteTensor(t1);                            // need to delete all the three tensors to avoid memory leak
  TF_DeleteTensor(t2);
  TF_DeleteTensor(tout);
  writeln('Finished Example 2');
  writeln;
  end;

procedure Example3;
// In this example two identical size matrices are multiplied element-wise. It uses the short version of the Eager type General
// interface (ExecOper)
  var
    t1:TF_TensorPtr;
    t2:TF_TensorPtr;
    attr:TF_DataType;
    tout:TF_TensorPtr;
  begin
  writeln('Starting Example 3');
  writeln;
  t1:=CreateTensorSingle([2,3],[1.0,2.0,3.0,4.0,5.0,6.0]);     // A 2x3 matrix
  t2:=CreateTensorSingle([2,3],[1.1,2.2,3.3,4.4,5.5,6.6]);     // Must be the same Shape
  attr:=TF_TensorType(t1);
  tout:=ExecOper('Mul',[t1,t2],['T'],['type'],[@attr],[true,true]);
  PrintTensorShape(tout,'TOut');
  PrintTensorData(tout);
  TF_DeleteTensor(tout); // only tout has to be deleted, t1 and t2 were delete in the ExecOper
  writeln('Finished Example 3');
  writeln;
  end;

procedure Example4;
// In this example two identical size matrices are added element-wise. It uses the Graph based specific interface (ExecAdd)
  var
    t1:TF_TensorPtr;
    t2:TF_TensorPtr;
    g:TGraphExt;       // The TGraphExt object that includes the detailed calls
    s:TSession;        // The TSession object we use to run
    touts:TF_TensorPtrs; // The result of TSession.Run in this case is an array as the output is given as an array as well
  begin
  writeln('Starting Example 4');
  writeln;
  t1:=CreateTensorSingle([2,3],[1.0,2.0,3.0,4.0,5.0,6.0]);     // A 2x3 matrix
  t2:=CreateTensorSingle([2,3],[1.1,2.2,3.3,4.4,5.5,6.6]);     // Must be the same Shape
  g.Init;                                         // Need to call Init before the first use
  g.AddInput('tensor1',TF_FLOAT);                 // not checking the result this time
  g.AddInput('tensor2',TF_FLOAT);
  g.AddAdd('tensor1','tensor2','tensorout',TF_FLOAT);
  s.Init(g);                                      // Need to make a Session that runs the Graph
  touts:=s.run(['tensor1','tensor2'],[t1,t2],['tensorout']); // The actual run of the Session
  PrintTensorShape(touts[0],'TOut[0]');           // A quick control that the product matrix is indeed 3x4, Int32
  PrintTensorData(touts[0]);
  s.Done;                                         // Release the Session and in it the SessionPtr
  g.Done;                                         // Release the Graph and in it the GraphPtr, the StatusPtr and the Outputs
  TF_DeleteTensor(t1);                            // need to delete all the three tensors to avoid memory leak
  TF_DeleteTensor(t2);
  TF_DeleteTensor(touts[0]);
  SetLength(touts,0);
  writeln('Finished Example 4');
  writeln;
  end;

procedure Example5;
// In this example the absolute value of one matrix is calculated (element-wise). It uses the Eager Specific interface (ExecAbs)
  var
    t:TF_TensorPtr;
    tout:TF_TensorPtr;
  begin
  writeln('Starting Example 5');
  writeln;
  t:=CreateTensorSingle([2,3],[-1.0,2.0,-3.0,4.0,-5.0,6.0]);
  tout:=ExecAbs(t);
  PrintTensorShape(tout);
  PrintTensorData(tout);
  TF_DeleteTensor(t);    // need to delete manually, because ExecAbs was called without adding ",true" to the end of the call
  TF_DeleteTensor(tout); // otherwise only tout should be deleted
  writeln('Finished Example 5');
  writeln;
  end;

procedure Example6;
// In this example a three step flow is implemented. In the first step one matrix (actually a vector) is substracted from another
// (same shape) matrix (element-wise).
// In the second step the square of the difference matrix is calculated (again element-wise).
// Finally in the third step the sum of the squares is calculated.
// This is basically a chi-square like approach (with no weighting).
  var
    t1,t2:TF_TensorPtr;
    g:TGraph;          // The TGraph object we will use
    s:TSession;        // The TSession object we use to run
    attr:TF_DataType;  // For the Generic AddOper all Attrubute Values are handed over as pointer and so a variable is needed
    tout:TF_TensorPtr;
  begin
  writeln('Starting Example 6');
  writeln;
  t1:=CreateTensorSingle([5],[-1.0,2.0,-3.0,4.0,-5.0]);
  t2:=CreateTensorSingle([5],[1.0,-2.0,3.0,-4.0,5.0]);
  attr:=TF_TensorType(t1);                        // Set the Attribute Value for Attribute "T"
  g.Init;
  g.AddInput('tensor1',TF_FLOAT);                 // Using the simple input adding method
  g.AddInput('tensor2',TF_FLOAT);
  g.AddOper('Sub',['tensor1','tensor2'],[],['difference'],['T'],['type'],[@attr]); // The first step
  g.AddOper('Square',['difference'],[],['squareofdifference'],['T'],['type'],[@attr]); // The second step
  g.AddConstant('zerodimension',Int32(0));               // need to add a constant tensor that is used to indicate which dimension to Sum
  g.AddOper('Sum',['squareofdifference','zerodimension'],[],['finalresult'],['T'],['type'],[@attr]); // The third step
  s.Init(g);                                      // Need to make a Session that runs the Graph
  tout:=s.run(['tensor1','tensor2'],[t1,t2],'finalresult'); // The actual run of the Session
  PrintTensorShape(tout,'TOut');                  // A quick control that the product matrix is indeed scalar
  PrintTensorData(tout);                          // Print the value (must be 220 = 2^2 + 4^2 + 6^2 + 8^2 + 10^2)
  s.Done;                                         // Release the Session and in it the SessionPtr
  g.Done;                                         // Release the Graph and in it the GraphPtr, the StatusPtr and the Outputs
  TF_DeleteTensor(t1);                            // need to delete all the three tensors to avoid memory leak
  TF_DeleteTensor(t2);
  TF_DeleteTensor(tout);
  writeln('Finished Example 6');
  writeln;
  end;

procedure Example7;
// This example is the same as Example 6, but uses the Specific Graph methods from the tf_wrapper family
  var
    t1,t2:TF_TensorPtr;
    g:TGraphExt;
    s:TSession;
    tout:TF_TensorPtr;
  begin
  writeln('Starting Example 7');
  writeln;
  t1:=CreateTensorSingle([5],[-1.0,2.0,-3.0,4.0,-5.0]);
  t2:=CreateTensorSingle([5],[1.0,-2.0,3.0,-4.0,5.0]);
  g.Init;
  g.AddInput('tensor1',TF_FLOAT);
  g.AddInput('tensor2',TF_FLOAT);
  g.AddSub('tensor1','tensor2','difference',TF_FLOAT); // This is the Specific AddOper version for Sub - AddSub
  g.AddSquare('difference','squareofdifference',TF_FLOAT);
  g.AddConstant('zerodimension',Int32(0));
  g.AddSum('squareofdifference','zerodimension','finalresult',false,TF_FLOAT,TF_INT32);
  s.Init(g);
  tout:=s.run(['tensor1','tensor2'],[t1,t2],'finalresult');
  PrintTensorShape(tout,'TOut');
  PrintTensorData(tout);
  s.Done;
  g.Done;
  TF_DeleteTensor(t1);
  TF_DeleteTensor(t2);
  TF_DeleteTensor(tout);
  writeln('Finished Example 7');
  writeln;
  end;

procedure Example8;
// Still the same as 6 and 7, but using the Generic Eager method
  var
    t1,t2:TF_TensorPtr;
    attr:TF_DataType;
  begin
  writeln('Starting Example 8');
  writeln;
  t1:=CreateTensorSingle([5],[-1.0,2.0,-3.0,4.0,-5.0]);
  t2:=CreateTensorSingle([5],[1.0,-2.0,3.0,-4.0,5.0]);
  attr:=TF_TensorType(t1);
  t1:=ExecOper('Sub',[t1,t2],['T'],['type'],[@attr],[true,true]); // tensors not used later are deleted immediately
  t1:=ExecOper('Square',t1,['T'],['type'],[@attr],[true]);
  t2:=CreateTensorInt32(0);
  t1:=ExecOper('Sum',[t1,t2],['T'],['type'],[@attr],[true,true]);
  PrintTensorShape(t1,'T1');
  PrintTensorData(t1);
  TF_DeleteTensor(t1); // only this tensor is left
  writeln('Finished Example 8');
  writeln;
  end;

procedure Example9;
// Still the same as 6, 7 and 8, but using the Specific Eager methods
  var
    t:TF_TensorPtr;
  begin
  writeln('Starting Example 9');
  writeln;
  t:=CreateTensorSingle([5],[-1.0,2.0,-3.0,4.0,-5.0]);
  t:=ExecSub(t,CreateTensorSingle([5],[1.0,-2.0,3.0,-4.0,5.0]),true,true); // Showing that CreateTensor can be used in-line
  t:=ExecSquare(t,true);
  t:=ExecSum(t,CreateTensorInt32(0),false,true,true);
  PrintTensorShape(t,'T');
  PrintTensorData(t);
  TF_DeleteTensor(t);
  writeln('Finished Example 9');
  writeln;
  end;

procedure Example10;
// Still the same as 6, 7, 8 and 9, but showing how (theoretically) operations can be linked
// It is strictly an EXAMPLE, not a recommendation, how to use it.
  var
    t:TF_TensorPtr;
  begin
  writeln('Starting Example 10');
  writeln;
  t:=ExecSum(ExecSquare(
                 ExecSub(
                     CreateTensorSingle([5],[-1.0,2.0,-3.0,4.0,-5.0]),
                     CreateTensorSingle([5],[1.0,-2.0,3.0,-4.0,5.0]),
                     true,
                     true),
                 true),
             CreateTensorInt32(0),
             false,
             true,
             true);
  PrintTensorShape(t,'T');
  PrintTensorData(t);
  TF_DeleteTensor(t);
  writeln('Finished Example 10');
  writeln;
  end;

procedure Example11;
// Still the same (especially Example9), but using a specific function for SquaredDifference
  var
    t:TF_TensorPtr;
  begin
  writeln('Starting Example 11');
  writeln;
  t:=CreateTensorSingle([5],[-1.0,2.0,-3.0,4.0,-5.0]);
  t:=ExecSquaredDifference(t,CreateTensorSingle([5],[1.0,-2.0,3.0,-4.0,5.0]),true,true); // difference and square in one step
  t:=ExecSum(t,CreateTensorInt32(0),false,true,true);
  PrintTensorShape(t,'T');
  PrintTensorData(t);
  TF_DeleteTensor(t);
  writeln('Finished Example 11');
  writeln;
  end;

procedure Example12;
// A simple example, how to generate e.g. a Tensor, filled with random numbers in a range
  var
    t:TF_TensorPtr;
  begin
  writeln('Starting Example 12');
  writeln;
  t:=CreateTensorSingleRandom([3,4,5],-10,10);
  PrintTensorShape(t,'T'); // its Shape is 3 dimensional, so the printout is sequential
  PrintTensorData(t);
  TF_DeleteTensor(t);
  writeln('Finished Example 12');
  writeln;
  end;

procedure Example13;
// This example converts a BMP image into a JPEG image while its contrast can also be adjusted
  var
    g:TGraphExt;
    s:TSession;
    t:TF_TensorPtr;
    LastOperationName:string;
  begin
  writeln('Starting Example 13');
  writeln;
  g.Init;
  g.AddInput('input-bmp',TF_String); // The input bmp file name will be given as an input parameter
  g.AddConstant('jpeg-resolution',Int32(80));
  g.AddReadFile('input-bmp','bmp-content');
  g.AddDecodeBmp('bmp-content','decoded-image',3);
  g.AddCast('decoded-image','extended-image',TF_UINT8,TF_FLOAT, false);
  g.AddConstant('contrast-factor',1.0); // no change
  g.AddAdjustContrastV2('extended-image','contrast-factor','contrasted-image',TF_Float);
  g.AddCast('contrasted-image','backsized-image',TF_FLOAT,TF_UINT8,false);
  g.AddEncodeJpegVariableQuality('backsized-image','jpeg-resolution','jpeg-content');
  g.AddConstant('output-jpg','myoutput.jpg'); // The output jpg name will be given as a constant (just to illustrate the difference)
  LastOperationName:=g.AddWriteFile('output-jpg','jpeg-content');
  s.Init(g);
  t:=CreateTensorString('myinput.bmp'); // Since the input bmp is an input parameter, we have to create it
  s.run([LastOperationName],['input-bmp'],[t]); // The actual run of the Session, making sure that the last operation runs
  TF_DeleteTensor(t); // In Graph operation, there is no automatic tensor deletion, so it has to be done manually
  s.Done;
  g.Done;
  writeln('The BMP file is converted to a JPG file');
  writeln;
  writeln('Finished Example 13');
  writeln;
  end;

procedure Example14;
// In this example a batch matrix multiplication is done on one batch and one fixed Single Tensor
  var
    t1:TF_TensorPtr;
    t2:TF_TensorPtr;
    attr:TF_DataType;
    tout:TF_TensorPtr;
  begin
  writeln('Starting Example 14');
  writeln;
  t1:=CreateTensorSingle([3,1,2],[1.0,2.0,3.0,4.0,5.0,6.0]);     // A 3 pieces of 1x2 matrix (3 long batch)
  t2:=CreateTensorSingle([2,4],[1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8]); // A 2x4 matrix to use for all
  attr:=TF_TensorType(t1);
  tout:=ExecOper('BatchMatMulV2',[t1,t2],['T'],['type'],[@attr],[false,true]); // V2 supports batch vs. non-batch
  PrintTensorData(tout);
  TF_DeleteTensor(t1); // Only t1 since t2 is deleted in ExecOper
  TF_DeleteTensor(tout);
  writeln('Finished Example 14');
  writeln;
  end;

procedure Example15;
  var
    g:TGraphExt;
    s:TSession;
    t:TF_TensorPtr;
    SaveOperationName:string;
    attr:TF_TypeList;
    DataList:TF_StringList;
  begin
  // This example shows how to save one Tensors.
  writeln('Starting Example 15');
  writeln;

  // first a float
  g.Init;
  t:=CreateTensorString([1],['fancynamesingle']); // This is a stringlist in one Tensor, with the name(s) to be used for the saved tensors
  g.AddTensor('whatnametouse',t,true);                         // This is used as a constant inside the Graph
  g.AddConstant('whatfilename','test-float.tft');                    // The file name (another Constant)
  g.AddInput('thisistosave',TF_FLOAT);                         // It will be the one to save
  attr:=nil;
  SetLength(attr,1);                                           // Before FPC 3.2 Dynamic Arrays cannot be called with values
  attr[0]:=TF_FLOAT;                                           // The first Attribute is the first (and now only) Tensor's type
  DataList:=nil;
  SetLength(DataList,1);
  DataList[0]:='thisistosave';
  SaveOperationName:=g.AddSave('whatfilename','whatnametouse',DataList,attr);

  t:=CreateTensorSingle([3,1,2],[0.00,0.01,1.00,1.01,2.00,2.01]); // This will be saved
  s.init(g);
  s.run([SaveOperationName],['thisistosave'],[t]);
  s.Done;
  g.Done;
  PrintTensorShape(t, ' Saved float');
  PrintTensorData(t, ' Saved float');

  TF_DeleteTensor(t);

  // and the same for a string tensor
  g.Init;
  t:=CreateTensorString([1],['fancynamestring']); // This is a stringlist in one Tensor, with the name(s) to be used for the saved tensors
  g.AddTensor('whatnametouse',t,true);                         // This is used as a constant inside the Graph
  g.AddConstant('whatfilename','test-string.tft');                    // The file name (another Constant)
  g.AddInput('thisistosave',TF_STRING);                         // It will be the one to save
  SaveOperationName:=g.AddSave('whatfilename','whatnametouse',['thisistosave'],[TF_STRING]); // here use a simplified version

  t:=CreateTensorString([1, 2],['qwerty', 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ']); // This will be saved
  s.init(g);
  s.run([SaveOperationName],['thisistosave'],[t]);
  s.Done;
  g.Done;
  PrintTensorShape(t, ' Saved string');
  PrintTensorData(t, ' Saved string');
  TF_DeleteTensor(t);

  writeln('Finished Example 15');
  writeln;
  end;

procedure Example16;
  var
    g:TGraphExt;
    s:TSession;
    t:TF_TensorPtr;
    tout:TF_TensorPtr;
  begin
  // And how to read it back
  writeln('Starting Example 16');
  writeln;

  // the float
  g.Init;
  g.AddConstant('whatfilepattern','test-float.tft'); // The file pattern, in this case a direct name
  t:=CreateTensorString('fancynamesingle'); // This is a string with the name of the tensor to restore
  g.AddTensor('whattorestore',t,true);
  g.AddRestore('whatfilepattern','whattorestore','readtensor',TF_FLOAT,-1);
  s.init(g);
  tout:=s.run([],[],'readtensor');
  PrintTensorShape(tout, ' Restored float');
  PrintTensorData(tout, ' Restored float');
  TF_DeleteTensor(tout);
  s.Done;
  g.Done;

  // and the string
  g.Init;
  g.AddConstant('whatfilepattern','test-string.tft'); // The file pattern, in this case a direct name
  t:=CreateTensorString('fancynamestring'); // This is a string with the name of the tensor to restore
  g.AddTensor('whattorestore',t,true);
  g.AddRestore('whatfilepattern','whattorestore','readtensor',TF_STRING,-1);
  s.init(g);
  tout:=s.run([],[],'readtensor');
  PrintTensorShape(tout, ' Restored string');
  PrintTensorData(tout, ' Restored string');

  TF_DeleteTensor(tout);
  s.Done;
  g.Done;

  writeln('Finished Example 16');
  writeln;
  end;

procedure Example17;
  var
    g:TGraphExt;
    s:TSession;
    t:TF_TensorPtr;
    touts:TF_TensorPtrs; // showing again the s.run with multiple outputs, even if that "multiple" is just one
  begin
  // This is to test the Frac(x) functionality on two different ways. The second seems much faster
  writeln('Starting Example 17');
  writeln;
  writeln('Be patient, it might take few seconds to complete!');
  writeln;
  t:=CreateTensorSingleRandom([10000,10000],0,1000);

  g.Init;
  g.AddInput('x',TF_FLOAT);
  g.AddConstant('1',1.0);
  g.AddMod('x','1','xmod1',TF_FLOAT);
  s.init(g);
  Writeln('Before x mod 1 ',DateTimeToStr(Now));
  touts:=s.run(['x'],[t],['xmod1']);
  Writeln('After x mod 1 ',DateTimeToStr(Now));
  PrintTensorShape(touts[0]);
  TF_DeleteTensor(touts[0]);
  SetLength(touts,0);
  s.Done;
  g.Done;

  g.Init;
  g.AddInput('x',TF_FLOAT);
  g.AddFloor('x','xfloor',TF_FLOAT);
  g.AddSub('x','xfloor','xminusxfloor',TF_FLOAT);
  s.init(g);
  Writeln('Before x - xfloor ',DateTimeToStr(Now));
  touts:=s.run(['x'],[t],['xminusxfloor']);
  Writeln('After x - x floor ',DateTimeToStr(Now));
  PrintTensorShape(touts[0]);
  TF_DeleteTensor(touts[0]);
  SetLength(touts,0);
  s.Done;
  g.Done;

  TF_DeleteTensor(t);

  writeln('Finished Example 17');
  writeln;
  end;

procedure Example18;
  var
    g:TGraphExt;
    s:TSession;
    t,t1,t2:TF_TensorPtr;
    SaveOperationName:string;
    attr:TF_TypeList;
    DataList:TF_StringList;

    tout: TF_TensorPtr;
  begin
  // This example shows how to save multiple Tensors in one save and restore them one-by-one.
  writeln('Starting Example 18');
  writeln;

  // save a float and an integer tensor in the same step
  g.Init;
  t:=CreateTensorString([2],['fancynamesingle','fancynameint']); // This is a stringlist in one Tensor, with the name(s) to be used for the saved tensors
  g.AddTensor('whatnametouse',t,true);                         // This is used as a constant inside the Graph
  g.AddConstant('whatfilename','test-multi.tft');                    // The file name (another Constant)
  g.AddConstant('emptystring','');                             // No encryption (another Constant)
  g.AddInput('thisistosave1',TF_FLOAT);                        // It will be the first to save
  g.AddInput('thisistosave2',TF_INT32);                        // It will be the second to save
  attr:=nil;
  SetLength(attr,2);                                           // Before FPC 3.2 Dynamic Arrays cannot be called with values
  attr[0]:=TF_FLOAT;                                           // The first Attribute is the first Tensor's type
  attr[1]:=TF_INT32;                                           // The first Attribute is the second Tensor's type
  DataList:=nil;
  SetLength(DataList,2);
  DataList[0]:='thisistosave1';
  DataList[1]:='thisistosave2';
  SaveOperationName:=g.AddSave('whatfilename','whatnametouse',DataList,attr);

  t1:=CreateTensorSingle([3,1,2],[1.1,2.2,3.3,4.4,5.5,6.6]); // This will be saved as first
  t2:=CreateTensorInt32([8],[1,2,3,4,5,6,7,8]);              // This will be saved as second
  s.init(g);
  s.run([SaveOperationName],['thisistosave1', 'thisistosave2'],[t1, t2]);
  PrintTensorShape(t1, ' Saved float');
  PrintTensorData(t1, ' Saved float');
  PrintTensorShape(t2, ' Saved integer');
  PrintTensorData(t2, ' Saved integer');
  TF_DeleteTensor(t1);
  TF_DeleteTensor(t2);
  s.Done;
  g.Done;

  // and then first restore the float tensor (same as Example16, hence not put in another Example)
  g.Init;
  g.AddConstant('whatfilepattern','test-multi.tft'); // The file pattern, in this case a direct name
  t:=CreateTensorString('fancynamesingle'); // This is a string with the name of the tensor to restore
  g.AddTensor('whattorestore',t,true);
  g.AddRestore('whatfilepattern','whattorestore','readtensor',TF_FLOAT,-1);
  s.init(g);
  tout:=s.run([],[],'readtensor');
  PrintTensorShape(tout, 'Restored float');
  PrintTensorData(tout, 'Restored float');
  TF_DeleteTensor(tout);
  s.Done;
  g.Done;

  // and finally the integer one
  g.Init;
  g.AddConstant('whatfilepattern','test-multi.tft'); // The file pattern, in this case a direct name
  t:=CreateTensorString('fancynameint'); // This is a string with the name of the tensor to restore
  g.AddTensor('whattorestore',t,true);
  g.AddRestore('whatfilepattern','whattorestore','readtensor',TF_INT32,-1);
  s.init(g);
  tout:=s.run([],[],'readtensor');
  PrintTensorShape(tout, 'Restored integer');
  PrintTensorData(tout, 'Restored integer');
  TF_DeleteTensor(tout);
  s.Done;
  g.Done;

  writeln('Finished Example 18');
  writeln;
  end;

procedure Example19;
  var
    g:TGraphExt;
    s:TSession;
    tvar,talpha,tdelta,tout:TF_TensorPtr;
    assignop,applyop:string;
  begin
  // This example shows how to use a resource-typed variable in an operation
  writeln('Starting Example 19');
  writeln;

  tvar:=CreateTensorSingleRandom([10],5,10);
  PrintTensorData(tvar,'The initial value');
  tdelta:=CreateTensorSingleRandom([10],0,1);
  PrintTensorData(tdelta,'The delta');
  talpha:=CreateTensorSingle(0.1);
  PrintTensorData(talpha,'The alpha');

  g.init;

  g.AddInput('initialvalue',TF_FLOAT); // The three inputs used
  g.AddInput('alpha',TF_FLOAT);
  g.AddInput('delta',TF_FLOAT);
  g.AddVarhandleOp('varresource','','',TF_FLOAT,TF_Shape.Create(10),nil); // Creates a variable referred as a "resource"
  assignop:=g.AddAssignVariableOp('varresource','initialvalue',TF_FLOAT, false); // Initialise the variable with the input
  applyop:=g.AddResourceApplyGradientDescent('varresource','alpha','delta',TF_FLOAT,true);
  g.AddReadVariableOp('varresource','result',TF_FLOAT); // the output in a tensor form

  s.Init(g);
  s.run([assignop],['initialvalue'],[tvar]); // Initialization
  s.run([applyop],['alpha','delta'],[talpha,tdelta]); // Only execution
  tout:=s.run([],[],'result'); // Read result in a separate step. Read in the same step might give pre-update result
  PrintTensorData(tout,'The result');

  s.done;
  g.done;

  TF_DeleteTensor(tvar);
  TF_DeleteTensor(talpha);
  TF_DeleteTensor(tdelta);
  TF_DeleteTensor(tout);
  writeln('Finished Example 19');
  writeln;
  end;

procedure Example20;
  var
    g:TGraphExt;
    s:TSession;
    tvar,talpha,tdelta,tout:TF_TensorPtr;
    assignop,applyop:string;
  begin
  // The same as #19, but using the "old" ref type variable
  writeln('Starting Example 20');
  writeln;

  tvar:=CreateTensorSingleRandom([10],5,10);
  PrintTensorData(tvar,'The initial value');
  tdelta:=CreateTensorSingleRandom([10],0,1);
  PrintTensorData(tdelta,'The delta');
  talpha:=CreateTensorSingle(0.1);
  PrintTensorData(talpha,'The alpha');

  g.init;

  g.AddInput('initialvalue',TF_FLOAT); // The three inputs used
  g.AddInput('alpha',TF_FLOAT);
  g.AddInput('delta',TF_FLOAT);
  g.AddVariable('varref',TF_Shape.Create(10),TF_FLOAT,'',''); // Creates a variable referred as a "ref(erence)"
  assignop:=g.AddAssign('varref','initialvalue','varref',TF_FLOAT,false,true); // Initialise the variable with the input
  applyop:=g.AddApplyGradientDescent('varref','alpha','delta','varref2',TF_FLOAT,true);

  s.Init(g);
  s.run([assignop],['initialvalue'],[tvar]); // Initialization

  tout:=s.run([applyop],['alpha','delta'],[talpha,tdelta],'varref'); // Execution and read in one step (no parallel ReadVariableOp like above)
  PrintTensorData(tout,'The result with the operation given and using the variable name');
  TF_DeleteTensor(tout);

  tout:=s.run(['alpha','delta'],[talpha,tdelta],'varref2'); // Can be done using the output of applyop, so no need to name the operation
  PrintTensorData(tout,'The result without the operation given and using the copy of the variable name. Notice, it is already the second iteration.');
  TF_DeleteTensor(tout);

  s.done;
  g.done;

  TF_DeleteTensor(tvar);
  TF_DeleteTensor(talpha);
  TF_DeleteTensor(tdelta);
  writeln('Finished Example 20');
  writeln;
  end;

procedure Example21;
  var
    g:TGraphExt;
    graphdef:TF_BufferPtr;
    f:file;
  begin
  // Shows how to save a Graph from outside the Graph
  // The saved Graph can be retrieved, but be careful, as the TGraph.FOperOutputs is not populated, not all easy TSession.run can work
  writeln('Starting Example 21');
  writeln;
  // a very simple "Graph"
  g.init;
  g.AddInput('input',TF_FLOAT);
  g.AddSquare('input','output',TF_FLOAT);
  // get the GraphDef
  graphdef:=New(TF_BufferPtr);
  graphdef^.data:=nil;
  graphdef^.length:=0;
  graphdef^.data_deallocator:=nil;
  TF_GraphToGraphDef(g.FGraph,graphdef,g.FStatus);
  TF_CheckStatus(g.FStatus);
  // save the GraphDef
  assignfile(f,'mygraph.op');
  rewrite(f,1);
  blockwrite(f,graphdef^.data,graphdef^.length);
  closefile(f);
  // remove the GraphDef
  with graphdef^ do
    data_deallocator(data,length);
  // finish the Graph
  g.done;

  writeln('The Graph is saved to teh disk');
  writeln;

  writeln('Finished Example 21');
  writeln;
  end;

procedure Example22;
  var
    g:TGraphExt;
    s:TSession;
    // tfilename:TF_TensorPtr;
    touts:TF_TensorPtrs;
  begin
  // CSV
  writeln('Starting Example 22');
  writeln;

  // a very simple "Graph"
  g.init;
  g.AddInput('filename',TF_STRING);
  g.AddConstant('default1','default');
  g.AddConstant('default2',Int32(9992));
  g.AddConstant('default3',Int32(9993));
  // g.AddReadFile('filename','filecontent');
  // I would need here an elementary step to split the filecontent dimensionless string tensor into individual lines, i.e. a
  // 1 dimensional string tensor after removing the header. I would call it "splitcontent", what I now simulate hereby.
  // Question raised: https://discuss.tensorflow.org/t/missing-step-in-csv-reading-and-decoding/14312
  // If you can help me, how to do the actual split, please let me know.
  g.AddTensor('splitcontent',CreateTensorString([3],['a, 1, 2','b, 3, 4', 'c, 5, 6']), true);
  g.AddDecodeCSV('splitcontent',TF_StringList.Create('default1','default2','default3'),TF_StringList.Create('coloumn1','coloumn2','coloumn3'),TF_TypeList.Create(TF_STRING,TF_INT32,TF_INT32),',',true,'N/A',TF_IntList.Create(0,1,2));

  // tfilename:=CreateTensorString('test.csv');

  s.init(g);
  // touts:=s.run(['filename'],[tfilename],['coloumn1','coloumn3', 'splitcontent', 'filecontent']); // it would be if we read the file
  touts:=s.run([],[],['coloumn1','coloumn3', 'splitcontent']); // the simulated version
  PrintTensorShape(touts[0],'First column');
  PrintTensorData(touts[0],'First column');
  PrintTensorShape(touts[1],'Third column');
  PrintTensorData(touts[1],'Third column');
  PrintTensorShape(touts[2],'splitcontent');
  PrintTensorData(touts[2],'splitcontent');
  // PrintTensorShape(touts[2],'filecontent');
  // PrintTensorData(touts[2],'filecontent');
  // TF_DeleteTensor(tfilename);
  TF_DeleteTensors(touts);
  s.done;
  g.done;

  writeln('Finished Example 22');
  writeln;
  end;

procedure Example23;
  var
    tin:TF_TensorPtr;   // The two input tensors
    tshape:TF_TensorPtr;
    g:TGraphExt;          // The TGraph object we will use
    s:TSession;        // The TSession object we use to run
    touts:TF_TensorPtrs; // The result of TSession.Run
    tout:TF_TensorPtr; // The result of the Eager run
  begin
  // Split a Tensor into a list of tensors in a Graph and then extract one tensor in eager mode
  writeln('Starting Example 23');
  writeln;
  // the Tensor to be split
  tin:=CreateTensorSingle([3,1,2],[1.0,2.0,3.0,4.0,5.0,6.0]);     // A 3x1x2 tensor to be split to 3 pieces of 1x2 tensors
  tshape:=CreateTensorInt32([1,2]); // I am not sure why it is needed at all, since it is always the subset of the input
  g.Init;
  if g.AddInput('tensorin',TF_Float)='' then writeln('Error while adding tensorin');
  g.AddInput('tensorshape',TF_Int32); // no need to check if it is obviously right
  if g.AddTensorListFromTensor('tensorin','tensorshape','tensorout',TF_Float,TF_Int32)='' then // split the tensor into a list
    writeln('Something went wrong!');
  if g.AddTensorListElementShape('tensorout','checkshape',TF_Int32)='' then // get the shape back
    writeln('Something went wrong!');
  s.Init(g);                                      // Need to make a Session that runs the Graph
  touts:=s.run(['tensorin','tensorshape'],[tin,tshape],['checkshape','tensorout']); // The actual run of the Session
  // touts[0] is the shape identified in the second step
  PrintTensorShape(touts[0],'tOuts[0]');
  PrintTensorData(touts[0]);
  // touts[1] is the actual tensorlist, the #1 element is extracted in an eager mode
  tout := ExecTensorListGetItem(touts[1],CreateTensorInt32(2),CreateTensorInt32([1,2,3,4,5,100,1000]),TF_Float,false,true,true); // What is the third parameter for? Clearly not used.
  PrintTensorShape(tout,'tOut');
  PrintTensorData(tout);
  s.Done;                                         // Release the Session and in it the SessionPtr
  g.Done;                                         // Release the Graph and in it the GraphPtr, the StatusPtr and the Outputs
  TF_DeleteTensor(tin);                           // Need to delete all the four tensors to avoid memory leak
  TF_DeleteTensor(tshape);
  TF_DeleteTensor(tout);
  TF_DeleteTensors(touts);
  writeln('Finished Example 23');
  writeln;
  end;

procedure Example24;
  begin
  // Print the TensorFlow release in use
  writeln('Starting Example 24');
  writeln;
  writeln('TF version: ', TF_Version);
  writeln;
  writeln('Finished Example 24');
  writeln;
  end;

procedure Example25;
  var
    t: TF_TensorPtr;
  begin
  // Create and print various String tensors
  writeln('Starting Example 25');
  writeln;

  t:=CreateTensorString('short string single');
  PrintTensorShape(t, 'a short string in a dimensionless tensor');
  PrintTensorData(t, 'a short string in a dimensionless tensor');
  TF_DeleteTensor(t);
  writeln;

  t:=CreateTensorString('long string single, i.e. a string that is longer than 22 characters');
  PrintTensorShape(t, 'a long string in a dimensionless tensor');
  PrintTensorData(t, 'a long string in a dimensionless tensor');
  TF_DeleteTensor(t);
  writeln;

  t:=CreateTensorString([1],['tensor with only one element']);
  PrintTensorShape(t, 'tensor with only one element');
  PrintTensorData(t, 'tensor with only one element');
  TF_DeleteTensor(t);
  writeln;

  t:=CreateTensorString([2],['a vector with a short','and a long element, i.e. the second is longer than 22 characters again']);
  PrintTensorShape(t, 'a vector');
  PrintTensorData(t, 'a vector');
  TF_DeleteTensor(t);
  writeln;

  t:=CreateTensorString([2,2],['a real array','with four elements','stored 2x2','but displayed sequentially']);
  PrintTensorShape(t, 'a 2D array');
  PrintTensorData(t, 'a 2D array');
  TF_DeleteTensor(t);
  writeln;

  t:=CreateTensorString([3,3],['a 3x3 matrix','b','c','d','e','f','g','h','i']);
  PrintTensorShape(t, 'a larger 2D array');
  PrintTensorData(t, 'a larger 2D array');
  TF_DeleteTensor(t);
  writeln;

  t:=CreateTensorString([2,2,2],['a 3D matrix','b','c','d','e','f','g','h']);
  PrintTensorShape(t, 'a 3D array');
  PrintTensorData(t, 'a 3D array');
  TF_DeleteTensor(t);

  writeln('Finished Example 25');
  writeln;
  end;

begin
Example1;
Example2;
Example3;
Example4;
Example5;
Example6;
Example7;
Example8;
Example9;
Example10;
Example11;
Example12;
Example13;
Example14;
Example15;
Example16;
Example17;
Example18;
Example19;
Example20;
Example21;
Example22;
Example23;
Example24;
Example25;
Sleep(1000); // Lazarus output window is a bit slow, so, not to loose earlier writeln messages, need a bit of slow-down
end.

