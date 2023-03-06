unit tf_customops;

//**********************************************************************************************************************************
//
//  Pascal methods to create and manage TensorFlow based custom operations, built over the raw operations
//
//  Copyright: (C) 2023, Zsolt Szakaly
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
//**********************************************************************************************************************************
//
//  Description
//
//  There is a general purpose tf_operations.pas unit, that helps to create TF graphs or to execute in an eager mode TF raw
//  operations on tensors created and managed through the tf_tensors.pas unit. There is also a wrapper around each and every raw
//  operation, again both in graph and eager approach, as it is in the tf_wrapper.pas unit (or later might be renamed
//  tf_rawops.pas).
//
//  This unit is to include more advanced operations that are built over the raw operations. The actual list and functionality of
//  such new methods is really a work-in-progress process, adding new operations if and when I come across the need of one.
//  As much as possible I try to keep the naming of operations and their parameters in line with other bindings (e.g the Python
//  binding) of TensorFlow, however their is no guarantee at all that it will always be the case.
//
//  To use functionality from this unit in a graph mode, the graph must be defined as an instance of the TGraphCustom object or one
//  of its descendents if such descendent is made by the user. The actual functions follow the same naming conventions as used in
//  the raw operation wraper, i.e. Add<oper>.
//
//  To use functionality in an eager mode, various Exec<oper> functions and procedures are defined. In general the purpose is to
//  have every function available both in graph and eager modes, however it might not always be possible.
//  The actual implementation of the eager Exec<oper> routines might be made through individual Exec<oper> steps (what is sometimes
//  slow) or through one call to the related Add<oper> method (what can be complicated). The choice between these two approaches is
//  made on a case-by-case basis, and the actual implementation might even change later.
//
//**********************************************************************************************************************************
//
//  Change log:
//    06/03/2023 Initial version
//
//**********************************************************************************************************************************

interface

uses
  ctypes,
  tf_api,
  tf_operations,
  tf_wrapper;

type
  TGraphCustom = object(TGraphExt)
    function AddFrac(const I_x:string; const O_y:string; const A_T:TF_DataType; const AOperationName:string=''):string;
    function AddStandardDeviation(const I_input:string; const I_reduction_indices:string; const O_y:string;
                                  const A_keep_dims:boolean; const A_T:TF_DataType; const A_Tidx:TF_DataType;
                                  const AOperationName:string=''):string;
    function AddVariance(const I_input:string; const I_reduction_indices:string; const O_y:string; const A_keep_dims:boolean;
                         const A_T:TF_DataType; const A_Tidx:TF_DataType; const AOperationName:string=''):string;

    end;

function ExecFrac(const I_input: TF_TensorPtr; const D_input: boolean = false):TF_TensorPtr;
function ExecStandardDeviation(const I_input:TF_TensorPtr; const I_reduction_indices:TF_TensorPtr; const A_keep_dims:boolean; const D_input:boolean = false; const D_reduction_indices:boolean = false):TF_TensorPtr;
function ExecVariance(const I_input:TF_TensorPtr; const I_reduction_indices:TF_TensorPtr; const A_keep_dims:boolean; const D_input:boolean = false; const D_reduction_indices:boolean = false):TF_TensorPtr;

implementation

function TGraphCustom.AddFrac(const I_x:string; const O_y:string; const A_T:TF_DataType; const AOperationName:string=''):string;
  var
    temp : string;
  begin
  // Frac can be done three different ways (see Example17). This is the frac(-1.1) = 0.9 version!
  temp := AddFloor(I_x, '', A_T);
  result := AddSub(I_x, temp, O_y, A_T, AOperationName);
  end;
function TGraphCustom.AddStandardDeviation(const I_input:string; const I_reduction_indices:string; const O_y:string;
                                           const A_keep_dims:boolean; const A_T:TF_DataType; const A_Tidx:TF_DataType;
                                           const AOperationName:string=''):string;
  var
    temp : string;
  begin
  temp := AddVariance(I_input, I_reduction_indices, '', A_keep_dims, A_T, A_Tidx);
  result := AddSqrt(temp, O_y, A_T, AOperationName);
  end;
function TGraphCustom.AddVariance(const I_input:string; const I_reduction_indices:string; const O_y:string; const A_keep_dims:boolean;
  const A_T:TF_DataType; const A_Tidx:TF_DataType; const AOperationName:string=''):string;
  var
    temp : string;
  begin
  temp := AddMean(I_input, I_reduction_indices, '', A_keep_dims, A_T, A_Tidx);
  temp := AddSquaredDifference(I_input, temp, '', A_T);
  result := AddMean(temp, I_reduction_indices, O_y, A_keep_dims, A_T, A_Tidx, AOperationName);
  end;

function ExecFrac(const I_input: TF_TensorPtr; const D_input: boolean = false):TF_TensorPtr;
  var
    Graph : TGraphCustom;
    Session:TSession;
    op : string;
  begin
  // ExecFrac could be done two eager ways (see Example28)

  // eager x mod 1 (elegant but the slowest and for negative numbers frac(-1.1) = -0.1)
  (*
  result := ExecMod(I_input, CreateTensorSinlge(1), D_input, true);
  *)

  // eager x - floor(x) (elegant, but still has two graph calls inside and double memory need, with the intended frac(-1.1) = 0.9)
  (*
  result := ExecSub(I_input, ExecFloor(I_input, false), D_input, true);
  *)

  // but here the faster graph solution is used (also because here frac(-1.1) = 0.9)
  Graph.Init;
  Graph.AddInput('x', TF_TensorType(I_input));
  op := Graph.AddFrac('x','',TF_TensorType(I_input));
  Session.init(Graph);
  result := Session.Run(['x'],[I_input], op);
  Session.Done;
  Graph.Done;
  if D_input then
    TF_DeleteTensor(I_input);
  end;
function ExecStandardDeviation(const I_input:TF_TensorPtr; const I_reduction_indices:TF_TensorPtr; const A_keep_dims:boolean;
                               const D_input:boolean = false; const D_reduction_indices:boolean = false):TF_TensorPtr;
  var
    Graph : TGraphCustom;
    Session:TSession;
  begin
  Graph.Init;
  Graph.AddInput('x', TF_TensorType(I_input));
  Graph.AddInput('idx', TF_TensorType(I_reduction_indices));
  Graph.AddStandardDeviation('x', 'idx', 'stdev', A_keep_dims, TF_TensorType(I_input), TF_TensorType(I_reduction_indices));
  Session.init(Graph);
  result := Session.Run(['x', 'idx'],[I_input, I_reduction_indices], 'stdev');
  Session.Done;
  Graph.Done;
  if D_input then
    TF_DeleteTensor(I_input);
  if D_reduction_indices then
    TF_DeleteTensor(I_reduction_indices);
  end;
function ExecVariance(const I_input:TF_TensorPtr; const I_reduction_indices:TF_TensorPtr; const A_keep_dims:boolean;
                      const D_input:boolean = false; const D_reduction_indices:boolean = false):TF_TensorPtr;
  var
    Graph : TGraphCustom;
    Session:TSession;
  begin
  Graph.Init;
  Graph.AddInput('x', TF_TensorType(I_input));
  Graph.AddInput('idx', TF_TensorType(I_reduction_indices));
  Graph.AddVariance('x', 'idx', 'variance', A_keep_dims, TF_TensorType(I_input), TF_TensorType(I_reduction_indices));
  Session.init(Graph);
  result := Session.Run(['x', 'idx'],[I_input, I_reduction_indices], 'variance');
  Session.Done;
  Graph.Done;
  if D_input then
    TF_DeleteTensor(I_input);
  if D_reduction_indices then
    TF_DeleteTensor(I_reduction_indices);
  end;

end.

