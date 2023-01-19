unit tf_api;

//**********************************************************************************************************************************
//
//  Pascal interface to TensorFlow dynamic library
//
//  Copyright: (C) 2020-2023, Zsolt Szakaly
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
//  Credit: This unit is based on the c_api.inc created by Phil Hess, based on the C header file c_api.h of TensorFlow
//
//  Change log:
//    13/02/2020 Initial version
//    21/02/2020 TF_DataType changed to cint from clong, because the TypeList works on cint
//    20/10/2021 Removed types declared unnecessary, like charPtr and replaced it with more standard PChar
//    22/10/2021 TF_DeleteTensors added to delete an array of tensors in one step
//    17/01/2023 Update to reflect c_api.h change from 1.15 to 2.11
//               - Added TF_TensorFromProto
//               - Added TF_StringView
//               - Added TF_NewOperationLocked
//               - Added TF_FinishOperationLocked
//               - Added TF_OperationAllInputs
//               - Added TF_OperationGetNumAttrs
//               . Added TF_OperationGetAttrNameLength
//               - Added TF_OperationGetAttrName
//               - Added TF_UpdateEdge
//               - Added TF_RegisterFilesystemPlugin
//               - Removed TF_StringEncode
//               - Removed TF_StringDecode
//               - Removed TF_StringEncodedSize
//               Added all string related routines as per the new tf_string.h in 2.11
//               Complete some missing structure and routines
//               - Added TF_GetAllRegisteredKernels
//               - Added TF_GetRegisteredKernelsForOp
//               - Added TF_Server and the related routines
//               - Added TF_RegisterLogListener
//    19/01/2023 TF_Tstring length changed
//
//**********************************************************************************************************************************
//
//  Description
//
//  This interface is trying to implement the c api published by the TensorFlow team on GitHub (www.github.com/tensorflow)
//  Certain types were added to provide easier use in the additional units based on this one, as well as a function to check
//  status codes.
//
//**********************************************************************************************************************************

{$linklib tensorflow}

interface

uses
  ctypes;

type
  ppcint64 = ^pcint64;
  ppcint = ^pcint;

function TF_Version: PChar; cdecl; external;

const

  TF_FLOAT = 1;
  TF_DOUBLE = 2;
  TF_INT32 = 3;
  TF_UINT8 = 4;
  TF_INT16 = 5;
  TF_INT8 = 6;
  TF_STRING = 7;
  TF_COMPLEX64 = 8;
  TF_COMPLEX = 8;
  TF_INT64 = 9;
  TF_BOOL = 10;
  TF_QINT8 = 11;
  TF_QUINT8 = 12;
  TF_QINT32 = 13;
  TF_BFLOAT16 = 14;
  TF_QINT16 = 15;
  TF_QUINT16 = 16;
  TF_UINT16 = 17;
  TF_COMPLEX128 = 18;
  TF_HALF = 19;
  TF_RESOURCE = 20;
  TF_VARIANT = 21;
  TF_UINT32 = 22;
  TF_UINT64 = 23;
type
  TF_DataType = cint;
  TF_DataTypePtr = ^TF_DataType;
function TF_DataTypeSize(dt: TF_DataType): csize_t; cdecl; external;

const
  TF_OK = 0;
  TF_CANCELLED = 1;
  TF_UNKNOWN = 2;
  TF_INVALID_ARGUMENT = 3;
  TF_DEADLINE_EXCEEDED = 4;
  TF_NOT_FOUND = 5;
  TF_ALREADY_EXISTS = 6;
  TF_PERMISSION_DENIED = 7;
  TF_UNAUTHENTICATED = 16;
  TF_RESOURCE_EXHAUSTED = 8;
  TF_FAILED_PRECONDITION = 9;
  TF_ABORTED = 10;
  TF_OUT_OF_RANGE = 11;
  TF_UNIMPLEMENTED = 12;
  TF_INTERNAL = 13;
  TF_UNAVAILABLE = 14;
  TF_DATA_LOSS = 15;
type
  TF_Code = clong;
  TF_Status = record end; // Opaque type
  TF_StatusPtr = ^TF_Status;
function TF_NewStatus: TF_StatusPtr; cdecl; external;
procedure TF_DeleteStatus(param1: TF_StatusPtr); cdecl; external;
procedure TF_SetStatus(s: TF_StatusPtr; code: TF_Code; msg: PChar); cdecl; external;
function TF_GetCode(s: TF_StatusPtr): TF_Code; cdecl; external;
function TF_Message(s: TF_StatusPtr): PChar; cdecl; external;
function TF_CheckStatus(AStatusPtr:TF_StatusPtr):Boolean;

type
  TF_Buffer = record
    data: pointer;
    length: csize_t;
    data_deallocator: procedure (data: pointer; length: csize_t); cdecl;
    end;
  TF_BufferPtr = ^TF_Buffer;
function TF_NewBufferFromString(proto: pointer; proto_len: csize_t): TF_BufferPtr; cdecl; external;
function TF_NewBuffer: TF_BufferPtr; cdecl; external;
procedure TF_DeleteBuffer(param1: TF_BufferPtr); cdecl; external;
function TF_GetBuffer(buffer: TF_BufferPtr): TF_Buffer; cdecl; external;

type
  TF_StringView = record
    data: PChar;
    len: csize_t;
    end;

type
  TF_Tensor = record end;
  TF_TensorPtr = ^TF_Tensor;
function TF_AllocateTensor(param1: TF_DataType; dims: pcint64; num_dims: cint; len: csize_t): TF_TensorPtr; cdecl; external;
function TF_TensorMaybeMove(tensor: TF_TensorPtr): TF_TensorPtr; cdecl; external;
procedure TF_DeleteTensor(param1: TF_TensorPtr); cdecl; external;
function TF_TensorType(param1: TF_TensorPtr): TF_DataType; cdecl; external;
function TF_NumDims(param1: TF_TensorPtr): cint; cdecl; external;
function TF_Dim(tensor: TF_TensorPtr; dim_index: cint): cint64; cdecl; external;
function TF_TensorByteSize(param1: TF_TensorPtr): csize_t; cdecl; external;
function TF_TensorData(param1: TF_TensorPtr): pointer; cdecl; external;
(* routines removed as they are not available after TF 2.4
function TF_StringEncode(src: PChar; src_len: csize_t; dst: PChar; dst_len: csize_t; status: TF_StatusPtr): csize_t; cdecl; external;
function TF_StringDecode(src: PChar; src_len: csize_t; dst: PPChar; dst_len: pcsize_t; status: TF_StatusPtr): csize_t; cdecl; external;
function TF_StringEncodedSize(len: csize_t): csize_t; cdecl; external;
*)

type
  TF_SessionOptions = record end;
  TF_SessionOptionsPtr = ^TF_SessionOptions;
function TF_NewSessionOptions: TF_SessionOptionsPtr; cdecl; external;
procedure TF_SetTarget(options: TF_SessionOptionsPtr; target: PChar); cdecl; external;
procedure TF_SetConfig(options: TF_SessionOptionsPtr; proto: pointer; proto_len: csize_t; status: TF_StatusPtr); cdecl; external;
procedure TF_DeleteSessionOptions(param1: TF_SessionOptionsPtr); cdecl; external;

type
  TF_Graph = record end;
  TF_GraphPtr = ^TF_Graph;
function TF_NewGraph: TF_GraphPtr; cdecl; external;
procedure TF_DeleteGraph(param1: TF_GraphPtr); cdecl; external;

type
  TF_OperationDescription = record end;
  TF_OperationDescriptionPtr = ^TF_OperationDescription;
  TF_Operation = record end;
  TF_OperationPtr = ^TF_Operation;
  TF_Input = record
    oper: TF_OperationPtr;
    index: cint;
    end;
  TF_InputPtr = ^TF_Input;
  TF_Output = record
    oper: TF_OperationPtr;
    index: cint;
    end;
  TF_Outputs = array of TF_Output;
  TF_OutputPtr = ^TF_Output;
  TF_Function = record end;
  TF_FunctionPtr = ^TF_Function;
  TF_FunctionOptions = record end;
  TF_FunctionOptionsPtr = ^TF_FunctionOptions;
procedure TF_GraphSetTensorShape(graph: TF_GraphPtr; output: TF_Output; dims: pcint64; num_dims: cint; status: TF_StatusPtr); cdecl; external;
function TF_GraphGetTensorNumDims(graph: TF_GraphPtr; output: TF_Output; status: TF_StatusPtr): cint; cdecl; external;
procedure TF_GraphGetTensorShape(graph: TF_GraphPtr; output: TF_Output; dims: pcint64; num_dims: cint; status: TF_StatusPtr); cdecl; external;
function TF_NewOperationLocked(graph: TF_GraphPtr; op_type: PChar; oper_name: PChar): TF_OperationDescriptionPtr; cdecl; external;
function TF_NewOperation(graph: TF_GraphPtr; op_type: PChar; oper_name: PChar): TF_OperationDescriptionPtr; cdecl; external;
procedure TF_SetDevice(desc: TF_OperationDescriptionPtr; device: PChar); cdecl; external;
procedure TF_AddInput(desc: TF_OperationDescriptionPtr; input: TF_Output); cdecl; external;
procedure TF_AddInputList(desc: TF_OperationDescriptionPtr; inputs: TF_OutputPtr; num_inputs: cint); cdecl; external;
procedure TF_AddControlInput(desc: TF_OperationDescriptionPtr; input: TF_OperationPtr); cdecl; external;
procedure TF_ColocateWith(desc: TF_OperationDescriptionPtr; op: TF_OperationPtr); cdecl; external;
procedure TF_SetAttrString(desc: TF_OperationDescriptionPtr; attr_name: PChar; value: pointer; length: csize_t); cdecl; external;
procedure TF_SetAttrStringList(desc: TF_OperationDescriptionPtr; attr_name: PChar; values: ppointer; lengths: pcsize_t; num_values: cint); cdecl; external;
procedure TF_SetAttrInt(desc: TF_OperationDescriptionPtr; attr_name: PChar; value: cint64); cdecl; external;
procedure TF_SetAttrIntList(desc: TF_OperationDescriptionPtr; attr_name: PChar; values: pcint64; num_values: cint); cdecl; external;
procedure TF_SetAttrFloat(desc: TF_OperationDescriptionPtr; attr_name: PChar; value: cfloat); cdecl; external;
procedure TF_SetAttrFloatList(desc: TF_OperationDescriptionPtr; attr_name: PChar; values: pcfloat; num_values: cint); cdecl; external;
procedure TF_SetAttrBool(desc: TF_OperationDescriptionPtr; attr_name: PChar; value: char); cdecl; external;
procedure TF_SetAttrBoolList(desc: TF_OperationDescriptionPtr; attr_name: PChar; values: PChar; num_values: cint); cdecl; external;
procedure TF_SetAttrType(desc: TF_OperationDescriptionPtr; attr_name: PChar; value: TF_DataType); cdecl; external;
procedure TF_SetAttrTypeList(desc: TF_OperationDescriptionPtr; attr_name: PChar; values: TF_DataTypePtr; num_values: cint); cdecl; external;
procedure TF_SetAttrFuncName(desc: TF_OperationDescriptionPtr; attr_name: PChar; value: PChar; length: csize_t); cdecl; external;
procedure TF_SetAttrShape(desc: TF_OperationDescriptionPtr; attr_name: PChar; dims: pcint64; num_dims: cint); cdecl; external;
procedure TF_SetAttrShapeList(desc: TF_OperationDescriptionPtr; attr_name: PChar; dims: ppcint64; num_dims: pcint; num_shapes: cint); cdecl; external;
procedure TF_SetAttrTensorShapeProto(desc: TF_OperationDescriptionPtr; attr_name: PChar; proto: pointer; proto_len: csize_t; status: TF_StatusPtr); cdecl; external;
procedure TF_SetAttrTensorShapeProtoList(desc: TF_OperationDescriptionPtr; attr_name: PChar; protos: ppointer; proto_lens: pcsize_t; num_shapes: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_SetAttrTensor(desc: TF_OperationDescriptionPtr; attr_name: PChar; value: TF_TensorPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_SetAttrTensorList(desc: TF_OperationDescriptionPtr; attr_name: PChar; values: TF_TensorPtr; num_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_SetAttrValueProto(desc: TF_OperationDescriptionPtr; attr_name: PChar; proto: pointer; proto_len: csize_t; status: TF_StatusPtr); cdecl; external;
function TF_FinishOperationLocked(desc: TF_OperationDescriptionPtr; status: TF_StatusPtr): TF_OperationPtr; cdecl; external;
function TF_FinishOperation(desc: TF_OperationDescriptionPtr; status: TF_StatusPtr): TF_OperationPtr; cdecl; external;
function TF_OperationName(oper: TF_OperationPtr): PChar; cdecl; external;
function TF_OperationOpType(oper: TF_OperationPtr): PChar; cdecl; external;
function TF_OperationDevice(oper: TF_OperationPtr): PChar; cdecl; external;
function TF_OperationNumOutputs(oper: TF_OperationPtr): cint; cdecl; external;
function TF_OperationOutputType(oper_out: TF_Output): TF_DataType; cdecl; external;
function TF_OperationOutputListLength(oper: TF_OperationPtr; arg_name: PChar; status: TF_StatusPtr): cint; cdecl; external;
function TF_OperationNumInputs(oper: TF_OperationPtr): cint; cdecl; external;
function TF_OperationInputType(oper_in: TF_Input): TF_DataType; cdecl; external;
function TF_OperationInputListLength(oper: TF_OperationPtr; arg_name: PChar; status: TF_StatusPtr): cint; cdecl; external;
function TF_OperationInput(oper_in: TF_Input): TF_Output; cdecl; external;
procedure TF_OperationAllInputs(oper: TF_OperationPtr; inputs: TF_OutputPtr; max_inputs: cint); cdecl; external;
function TF_OperationOutputNumConsumers(oper_out: TF_Output): cint; cdecl; external;
function TF_OperationOutputConsumers(oper_out: TF_Output; consumers: TF_InputPtr; max_consumers: cint): cint; cdecl; external;
function TF_OperationNumControlInputs(oper: TF_OperationPtr): cint; cdecl; external;
function TF_OperationGetControlInputs(oper: TF_OperationPtr; control_inputs: TF_OperationPtr; max_control_inputs: cint): cint; cdecl; external;
function TF_OperationNumControlOutputs(oper: TF_OperationPtr): cint; cdecl; external;
function TF_OperationGetControlOutputs(oper: TF_OperationPtr; control_outputs: TF_OperationPtr; max_control_outputs: cint): cint; cdecl; external;

const
  TF_ATTR_STRING = 0;
  TF_ATTR_INT = 1;
  TF_ATTR_FLOAT = 2;
  TF_ATTR_BOOL = 3;
  TF_ATTR_TYPE = 4;
  TF_ATTR_SHAPE = 5;
  TF_ATTR_TENSOR = 6;
  TF_ATTR_PLACEHOLDER = 7;
  TF_ATTR_FUNC = 8;
type
  TF_AttrType = clong;
  TF_AttrMetadata = record
    is_list: char;
    list_size: cint64;
    type_: TF_AttrType;
    total_size: cint64;
    end;
  TF_AttrMetadataPtr = ^TF_AttrMetadata;
function TF_OperationGetAttrMetadata(oper: TF_OperationPtr; attr_name: PChar; status: TF_StatusPtr): TF_AttrMetadata; cdecl; external;
procedure TF_OperationGetAttrString(oper: TF_OperationPtr; attr_name: PChar; value: pointer; max_length: csize_t; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrStringList(oper: TF_OperationPtr; attr_name: PChar; values: ppointer; lengths: pcsize_t; max_values: cint; storage: pointer; storage_size: csize_t; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrInt(oper: TF_OperationPtr; attr_name: PChar; value: pcint64; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrIntList(oper: TF_OperationPtr; attr_name: PChar; values: pcint64; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrFloat(oper: TF_OperationPtr; attr_name: PChar; value: pcfloat; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrFloatList(oper: TF_OperationPtr; attr_name: PChar; values: pcfloat; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrBool(oper: TF_OperationPtr; attr_name: PChar; value: PChar; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrBoolList(oper: TF_OperationPtr; attr_name: PChar; values: PChar; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrType(oper: TF_OperationPtr; attr_name: PChar; value: TF_DataTypePtr; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrTypeList(oper: TF_OperationPtr; attr_name: PChar; values: TF_DataTypePtr; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrShape(oper: TF_OperationPtr; attr_name: PChar; value: pcint64; num_dims: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrShapeList(oper: TF_OperationPtr; attr_name: PChar; dims: ppcint64; num_dims: pcint; num_shapes: cint; storage: pcint64; storage_size: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrTensorShapeProto(oper: TF_OperationPtr; attr_name: PChar; value: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrTensorShapeProtoList(oper: TF_OperationPtr; attr_name: PChar; values: TF_BufferPtr; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrTensor(oper: TF_OperationPtr; attr_name: PChar; value: TF_TensorPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrTensorList(oper: TF_OperationPtr; attr_name: PChar; values: TF_TensorPtr; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrValueProto(oper: TF_OperationPtr; attr_name: PChar; output_attr_value: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
function TF_OperationGetNumAttrs(oper: TF_OperationPtr): cint; cdecl; external;
function TF_OperationGetAttrNameLength(oper: TF_OperationPtr; i: cint): cint; cdecl; external;
procedure TF_OperationGetAttrName(oper: TF_OperationPtr; i: cint; output: PChar; status: TF_StatusPtr); cdecl; external;
function TF_GraphOperationByName(graph: TF_GraphPtr; oper_name: PChar): TF_OperationPtr; cdecl; external;
function TF_GraphNextOperation(graph: TF_GraphPtr; pos: pcsize_t): TF_OperationPtr; cdecl; external;
procedure TF_GraphToGraphDef(graph: TF_GraphPtr; output_graph_def: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_GraphGetOpDef(graph: TF_GraphPtr; op_name: PChar; output_op_def: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_GraphVersions(graph: TF_GraphPtr; output_version_def: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;

type
  TF_ImportGraphDefOptions = record end;
  TF_ImportGraphDefOptionsPtr = ^TF_ImportGraphDefOptions;
function TF_NewImportGraphDefOptions: TF_ImportGraphDefOptionsPtr; cdecl; external;
procedure TF_DeleteImportGraphDefOptions(opts: TF_ImportGraphDefOptionsPtr); cdecl; external;
procedure TF_ImportGraphDefOptionsSetPrefix(opts: TF_ImportGraphDefOptionsPtr; prefix: PChar); cdecl; external;
procedure TF_ImportGraphDefOptionsSetUniquifyNames(opts: TF_ImportGraphDefOptionsPtr; uniquify_names: char); cdecl; external;
procedure TF_ImportGraphDefOptionsSetUniquifyPrefix(opts: TF_ImportGraphDefOptionsPtr; uniquify_prefix: char); cdecl; external;
procedure TF_ImportGraphDefOptionsAddInputMapping(opts: TF_ImportGraphDefOptionsPtr; src_name: PChar; src_index: cint; dst: TF_Output); cdecl; external;
procedure TF_ImportGraphDefOptionsRemapControlDependency(opts: TF_ImportGraphDefOptionsPtr; src_name: PChar; dst: TF_OperationPtr); cdecl; external;
procedure TF_ImportGraphDefOptionsAddControlDependency(opts: TF_ImportGraphDefOptionsPtr; oper: TF_OperationPtr); cdecl; external;
procedure TF_ImportGraphDefOptionsAddReturnOutput(opts: TF_ImportGraphDefOptionsPtr; oper_name: PChar; index: cint); cdecl; external;
function TF_ImportGraphDefOptionsNumReturnOutputs(opts: TF_ImportGraphDefOptionsPtr): cint; cdecl; external;
procedure TF_ImportGraphDefOptionsAddReturnOperation(opts: TF_ImportGraphDefOptionsPtr; oper_name: PChar); cdecl; external;
function TF_ImportGraphDefOptionsNumReturnOperations(opts: TF_ImportGraphDefOptionsPtr): cint; cdecl; external;

type
  TF_ImportGraphDefResults = record end;
  TF_ImportGraphDefResultsPtr = ^TF_ImportGraphDefResults;
procedure TF_ImportGraphDefResultsReturnOutputs(results: TF_ImportGraphDefResultsPtr; num_outputs: pcint; outputs: TF_OutputPtr); cdecl; external;
procedure TF_ImportGraphDefResultsReturnOperations(results: TF_ImportGraphDefResultsPtr; num_opers: pcint; opers: TF_OperationPtr); cdecl; external;
procedure TF_ImportGraphDefResultsMissingUnusedInputMappings(results: TF_ImportGraphDefResultsPtr; num_missing_unused_input_mappings: pcint; src_names: PPPChar; src_indexes: ppcint); cdecl; external;
procedure TF_DeleteImportGraphDefResults(results: TF_ImportGraphDefResultsPtr); cdecl; external;
function TF_GraphImportGraphDefWithResults(graph: TF_GraphPtr; graph_def: TF_BufferPtr; options: TF_ImportGraphDefOptionsPtr; status: TF_StatusPtr): TF_ImportGraphDefResultsPtr; cdecl; external;
procedure TF_GraphImportGraphDefWithReturnOutputs(graph: TF_GraphPtr; graph_def: TF_BufferPtr; options: TF_ImportGraphDefOptionsPtr; return_outputs: TF_OutputPtr; num_return_outputs: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_GraphImportGraphDef(graph: TF_GraphPtr; graph_def: TF_BufferPtr; options: TF_ImportGraphDefOptionsPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_GraphCopyFunction(g: TF_GraphPtr; func: TF_FunctionPtr; grad: TF_FunctionPtr; status: TF_StatusPtr); cdecl; external;
function TF_GraphNumFunctions(g: TF_GraphPtr): cint; cdecl; external;
function TF_GraphGetFunctions(g: TF_GraphPtr; funcs: TF_FunctionPtr; max_func: cint; status: TF_StatusPtr): cint; cdecl; external;
procedure TF_OperationToNodeDef(oper: TF_OperationPtr; output_node_def: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;

type
  TF_WhileParams = record
    ninputs: cint;
    cond_graph: TF_GraphPtr;
    cond_inputs: TF_OutputPtr;
    cond_output: TF_Output;
    body_graph: TF_GraphPtr;
    body_inputs: TF_OutputPtr;
    body_outputs: TF_OutputPtr;
    name: PChar;
    end;
  TF_WhileParamsPtr = ^TF_WhileParams;
function TF_NewWhile(g: TF_GraphPtr; inputs: TF_OutputPtr; ninputs: cint; status: TF_StatusPtr): TF_WhileParams; cdecl; external;
procedure TF_FinishWhile(params: TF_WhileParamsPtr; status: TF_StatusPtr; outputs: TF_OutputPtr); cdecl; external;
procedure TF_AbortWhile(params: TF_WhileParamsPtr); cdecl; external;
procedure TF_AddGradients(g: TF_GraphPtr; y: TF_OutputPtr; ny: cint; x: TF_OutputPtr; nx: cint; dx: TF_OutputPtr; status: TF_StatusPtr; dy: TF_OutputPtr); cdecl; external;
function TF_GraphToFunction(fn_body: TF_GraphPtr; fn_name: PChar; append_hash_to_fn_name: char; num_opers: cint; opers: TF_OperationPtr; ninputs: cint; inputs: TF_OutputPtr; noutputs: cint; outputs: TF_OutputPtr; output_names: PPChar; opts: TF_FunctionOptionsPtr; description: PChar; status: TF_StatusPtr): TF_FunctionPtr; cdecl; external;
procedure TF_FunctionToFunctionDef(func: TF_FunctionPtr; output_func_def: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
function TF_FunctionImportFunctionDef(proto: pointer; proto_len: csize_t; status: TF_StatusPtr): TF_FunctionPtr; cdecl; external;
procedure TF_FunctionSetAttrValueProto(func: TF_FunctionPtr; attr_name: PChar; proto: pointer; proto_len: csize_t; status: TF_StatusPtr); cdecl; external;
procedure TF_FunctionGetAttrValueProto(func: TF_FunctionPtr; attr_name: PChar; output_attr_value: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_DeleteFunction(func: TF_FunctionPtr); cdecl; external;
function TF_TryEvaluateConstant(graph: TF_GraphPtr; output: TF_Output; result_: TF_TensorPtr; status: TF_StatusPtr): char; cdecl; external;

type
  TF_Session = record end;
  TF_SessionPtr = ^TF_Session;
function TF_NewSession(graph: TF_GraphPtr; opts: TF_SessionOptionsPtr; status: TF_StatusPtr): TF_SessionPtr; cdecl; external;
function TF_LoadSessionFromSavedModel(session_options: TF_SessionOptionsPtr; run_options: TF_BufferPtr; export_dir: PChar; tags: PPChar; tags_len: cint; graph: TF_GraphPtr; meta_graph_def: TF_BufferPtr; status: TF_StatusPtr): TF_SessionPtr; cdecl; external;
procedure TF_CloseSession(param1: TF_SessionPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_DeleteSession(param1: TF_SessionPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_SessionRun(session: TF_SessionPtr; run_options: TF_BufferPtr; inputs: TF_OutputPtr; input_values: TF_TensorPtr; ninputs: cint; outputs: TF_OutputPtr; output_values: TF_TensorPtr; noutputs: cint; target_opers: TF_OperationPtr; ntargets: cint; run_metadata: TF_BufferPtr; param12: TF_StatusPtr); cdecl; external;
procedure TF_SessionPRunSetup(param1: TF_SessionPtr; inputs: TF_OutputPtr; ninputs: cint; outputs: TF_OutputPtr; noutputs: cint; target_opers: TF_OperationPtr; ntargets: cint; handle: PPChar; param9: TF_StatusPtr); cdecl; external;
procedure TF_SessionPRun(param1: TF_SessionPtr; handle: PChar; inputs: TF_OutputPtr; input_values: TF_TensorPtr; ninputs: cint; outputs: TF_OutputPtr; output_values: TF_TensorPtr; noutputs: cint; target_opers: TF_OperationPtr; ntargets: cint; param11: TF_StatusPtr); cdecl; external;
procedure TF_DeletePRunHandle(handle: PChar); cdecl; external;

type
  TF_DeprecatedSession = record end;
  TF_DeprecatedSessionPtr = ^TF_DeprecatedSession;
function TF_NewDeprecatedSession(param1: TF_SessionOptionsPtr; status: TF_StatusPtr): TF_DeprecatedSessionPtr; cdecl; external;
procedure TF_CloseDeprecatedSession(param1: TF_DeprecatedSessionPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_DeleteDeprecatedSession(param1: TF_DeprecatedSessionPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_Reset(opt: TF_SessionOptionsPtr; containers: PPChar; ncontainers: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_ExtendGraph(param1: TF_DeprecatedSessionPtr; proto: pointer; proto_len: csize_t; param4: TF_StatusPtr); cdecl; external;
procedure TF_Run(param1: TF_DeprecatedSessionPtr; run_options: TF_BufferPtr; input_names: PPChar; inputs: TF_TensorPtr; ninputs: cint; output_names: PPChar; outputs: TF_TensorPtr; noutputs: cint; target_oper_names: PPChar; ntargets: cint; run_metadata: TF_BufferPtr; param12: TF_StatusPtr); cdecl; external;
procedure TF_PRunSetup(param1: TF_DeprecatedSessionPtr; input_names: PPChar; ninputs: cint; output_names: PPChar; noutputs: cint; target_oper_names: PPChar; ntargets: cint; handle: PPChar; param9: TF_StatusPtr); cdecl; external;
procedure TF_PRun(param1: TF_DeprecatedSessionPtr; handle: PChar; input_names: PPChar; inputs: TF_TensorPtr; ninputs: cint; output_names: PPChar; outputs: TF_TensorPtr; noutputs: cint; target_oper_names: PPChar; ntargets: cint; param11: TF_StatusPtr); cdecl; external;

type
  TF_DeviceList = record end;
  TF_DeviceListPtr = ^TF_DeviceList;
function TF_SessionListDevices(session: TF_SessionPtr; status: TF_StatusPtr): TF_DeviceListPtr; cdecl; external;
function TF_DeprecatedSessionListDevices(session: TF_DeprecatedSessionPtr; status: TF_StatusPtr): TF_DeviceListPtr; cdecl; external;
procedure TF_DeleteDeviceList(list: TF_DeviceListPtr); cdecl; external;
function TF_DeviceListCount(list: TF_DeviceListPtr): cint; cdecl; external;
function TF_DeviceListName(list: TF_DeviceListPtr; index: cint; status: TF_StatusPtr): PChar; cdecl; external;
function TF_DeviceListType(list: TF_DeviceListPtr; index: cint; status: TF_StatusPtr): PChar; cdecl; external;
function TF_DeviceListMemoryBytes(list: TF_DeviceListPtr; index: cint; status: TF_StatusPtr): cint64; cdecl; external;

type
  TF_Library = record end;
  TF_LibraryPtr = ^TF_Library;
function TF_LoadLibrary(library_filename: PChar; status: TF_StatusPtr): TF_LibraryPtr; cdecl; external;
function TF_GetOpList(lib_handle: TF_LibraryPtr): TF_Buffer; cdecl; external;
procedure TF_DeleteLibraryHandle(lib_handle: TF_LibraryPtr); cdecl; external;
function TF_GetAllOpList: TF_BufferPtr; cdecl; external;

type
  TF_ApiDefMap = record end;
  TF_ApiDefMapPtr = ^TF_ApiDefMap;
function TF_NewApiDefMap(op_list_buffer: TF_BufferPtr; status: TF_StatusPtr): TF_ApiDefMapPtr; cdecl; external;
procedure TF_DeleteApiDefMap(apimap: TF_ApiDefMapPtr); cdecl; external;
procedure TF_ApiDefMapPut(api_def_map: TF_ApiDefMapPtr; text: PChar; text_len: csize_t; status: TF_StatusPtr); cdecl; external;
function TF_ApiDefMapGet(api_def_map: TF_ApiDefMapPtr; name: PChar; name_len: csize_t; status: TF_StatusPtr): TF_BufferPtr; cdecl; external;

function TF_GetAllRegisteredKernels(status: TF_StatusPtr): TF_BufferPtr; cdecl; external;
function TF_GetRegisteredKernelsForOp(name: PChar; status: TF_StatusPtr): TF_BufferPtr; cdecl; external;

procedure TF_UpdateEdge(graph: TF_GraphPtr; new_src: TF_Output; dst: TF_Input; status: TF_StatusPtr); cdecl; external;

type
  TF_Server = record end;
  TF_ServerPtr = ^TF_Server;

function TF_NewServer(proto: pointer; proto_len: csize_t; status: TF_StatusPtr): TF_ServerPtr; cdecl; external;
procedure TF_ServerStart(server: TF_ServerPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_ServerStop(server: TF_ServerPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_ServerJoin(server: TF_ServerPtr; status: TF_StatusPtr); cdecl; external;
function TF_ServerTarget(server: TF_ServerPtr): PChar; cdecl; external;
procedure TF_DeleteServer(server: TF_ServerPtr); cdecl; external;

type
  TF_ListenerProcedure = procedure(message: PChar);

procedure TF_RegisterLogListener(listener: TF_ListenerProcedure); cdecl; external;

procedure TF_RegisterFilesystemPlugin(plugin_filename: PChar; status: TF_StatusPtr); cdecl; external;

type
  TF_TString = record
    raw: packed array[1..24] of char;
    end;
  TF_TStringPtr = ^TF_TString;
  TF_TString_Type = byte;

procedure TF_StringInit(t: TF_TStringPtr); cdecl; external;
procedure TF_StringCopy(dst: TF_TstringPtr; src: PChar; size: csize_t); cdecl; external;
procedure TF_StringAssignView(dst: TF_TstringPtr; src: PChar; size: csize_t); cdecl; external;
function TF_StringGetDataPointer(tstr: TF_TStringPtr): PChar; cdecl; external;
function TF_StringGetType(str: TF_TStringPtr): TF_TString_Type; cdecl; external;
function TF_StringGetSize(tstr: TF_TStringPtr): csize_t; cdecl; external;
function TF_StringGetCapacity(str: TF_TStringPtr): csize_t; cdecl; external;
procedure TF_StringDealloc(tstr: TF_TStringPtr); cdecl; external;

type
  TF_Shape=array of Int64;
  TF_ShapePtr=^TF_Shape;
  TF_FloatList=array of cfloat;
  TF_FuncnameList=array of string;
  TF_IntList=array of cint64;
  TF_ShapeList=array of TF_Shape;
  TF_StringList=array of string;
  TF_TypeList=array of TF_DataType;
  TF_TensorPtrs = array of TF_TensorPtr;

procedure TF_TensorFromProto(aFrom: TF_BufferPtr; aTo: TF_TensorPtr; aStatus: TF_StatusPtr); cdecl; external;

procedure TF_DeleteTensors(ATensors: TF_TensorPtrs);

implementation

uses
  SysUtils;

function TF_CheckStatus(AStatusPtr:TF_StatusPtr):Boolean;
  begin
  result:=TF_GetCode(AStatusPtr)=TF_OK;
  if not result then
    raise exception.create(TF_Message(AStatusPtr));
  end;

procedure TF_DeleteTensors(ATensors: TF_TensorPtrs);
  var Tensor : TF_TensorPtr;
  begin
  for Tensor in ATensors do
    TF_DeleteTEnsor(Tensor);
  end;

end.

