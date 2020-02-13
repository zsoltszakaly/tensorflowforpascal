unit tf_api;

//**********************************************************************************************************************************
//
//  Pascal interface to TensorFlow dynamic library
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
//  Credit: This unit is based on the c_api.inc created by Phil Hess, based on the C header file c_api.h of TensorFlow
//
//  Change log: 13/02/2020 Initial version
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
  OpaqueType = record
    end;
  charPtr = PAnsiChar;
  charPtrPtr = ^charPtr;
  charPtrPtrPtr = ^charPtrPtr;
  cint64Ptr = ^cint64;
  cint64PtrPtr = ^cint64Ptr;
  csize_tPtr = ^csize_t;
  pointerPtr = ^Pointer;
  cfloatPtr = ^cfloat;
  cintPtr = ^cint;
  cintPtrPtr = ^cintPtr;
  TF_DataTypePtr = Pointer;

function TF_Version: charPtr; cdecl; external;

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
  TF_DataType = clong;
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
  TF_Status = OpaqueType;
  TF_StatusPtr = ^TF_Status;
function TF_NewStatus: TF_StatusPtr; cdecl; external;
procedure TF_DeleteStatus(param1: TF_StatusPtr); cdecl; external;
procedure TF_SetStatus(s: TF_StatusPtr; code: TF_Code; msg: charPtr); cdecl; external;
function TF_GetCode(s: TF_StatusPtr): TF_Code; cdecl; external;
function TF_Message(s: TF_StatusPtr): charPtr; cdecl; external;
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
  TF_Tensor = OpaqueType;
  TF_TensorPtr = ^TF_Tensor;
  TF_TensorPtrs = array of TF_TensorPtr;
function TF_AllocateTensor(param1: TF_DataType; dims: cint64Ptr; num_dims: cint; len: csize_t): TF_TensorPtr; cdecl; external;
function TF_TensorMaybeMove(tensor: TF_TensorPtr): TF_TensorPtr; cdecl; external;
procedure TF_DeleteTensor(param1: TF_TensorPtr); cdecl; external;
function TF_TensorType(param1: TF_TensorPtr): TF_DataType; cdecl; external;
function TF_NumDims(param1: TF_TensorPtr): cint; cdecl; external;
function TF_Dim(tensor: TF_TensorPtr; dim_index: cint): cint64; cdecl; external;
function TF_TensorByteSize(param1: TF_TensorPtr): csize_t; cdecl; external;
function TF_TensorData(param1: TF_TensorPtr): pointer; cdecl; external;
function TF_StringEncode(src: charPtr; src_len: csize_t; dst: charPtr; dst_len: csize_t; status: TF_StatusPtr): csize_t; cdecl; external;
function TF_StringDecode(src: charPtr; src_len: csize_t; dst: charPtrPtr; dst_len: csize_tPtr; status: TF_StatusPtr): csize_t; cdecl; external;
function TF_StringEncodedSize(len: csize_t): csize_t; cdecl; external;

type
  TF_SessionOptions = OpaqueType;
  TF_SessionOptionsPtr = ^TF_SessionOptions;
function TF_NewSessionOptions: TF_SessionOptionsPtr; cdecl; external;
procedure TF_SetTarget(options: TF_SessionOptionsPtr; target: charPtr); cdecl; external;
procedure TF_SetConfig(options: TF_SessionOptionsPtr; proto: pointer; proto_len: csize_t; status: TF_StatusPtr); cdecl; external;
procedure TF_DeleteSessionOptions(param1: TF_SessionOptionsPtr); cdecl; external;

type
  TF_Graph = OpaqueType;
  TF_GraphPtr = ^TF_Graph;
function TF_NewGraph: TF_GraphPtr; cdecl; external;
procedure TF_DeleteGraph(param1: TF_GraphPtr); cdecl; external;

type
  TF_OperationDescription = OpaqueType;
  TF_OperationDescriptionPtr = ^TF_OperationDescription;
  TF_Operation = OpaqueType;
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
  TF_OutputPtr = ^TF_Output;
  TF_Function = OpaqueType;
  TF_FunctionPtr = ^TF_Function;
  TF_FunctionOptions = OpaqueType;
  TF_FunctionOptionsPtr = ^TF_FunctionOptions;
procedure TF_GraphSetTensorShape(graph: TF_GraphPtr; output: TF_Output; dims: cint64Ptr; num_dims: cint; status: TF_StatusPtr); cdecl; external;
function TF_GraphGetTensorNumDims(graph: TF_GraphPtr; output: TF_Output; status: TF_StatusPtr): cint; cdecl; external;
procedure TF_GraphGetTensorShape(graph: TF_GraphPtr; output: TF_Output; dims: cint64Ptr; num_dims: cint; status: TF_StatusPtr); cdecl; external;
function TF_NewOperation(graph: TF_GraphPtr; op_type: charPtr; oper_name: charPtr): TF_OperationDescriptionPtr; cdecl; external;
procedure TF_SetDevice(desc: TF_OperationDescriptionPtr; device: charPtr); cdecl; external;
procedure TF_AddInput(desc: TF_OperationDescriptionPtr; input: TF_Output); cdecl; external;
procedure TF_AddInputList(desc: TF_OperationDescriptionPtr; inputs: TF_OutputPtr; num_inputs: cint); cdecl; external;
procedure TF_AddControlInput(desc: TF_OperationDescriptionPtr; input: TF_OperationPtr); cdecl; external;
procedure TF_ColocateWith(desc: TF_OperationDescriptionPtr; op: TF_OperationPtr); cdecl; external;
procedure TF_SetAttrString(desc: TF_OperationDescriptionPtr; attr_name: charPtr; value: pointer; length: csize_t); cdecl; external;
procedure TF_SetAttrStringList(desc: TF_OperationDescriptionPtr; attr_name: charPtr; values: pointerPtr; lengths: csize_tPtr; num_values: cint); cdecl; external;
procedure TF_SetAttrInt(desc: TF_OperationDescriptionPtr; attr_name: charPtr; value: cint64); cdecl; external;
procedure TF_SetAttrIntList(desc: TF_OperationDescriptionPtr; attr_name: charPtr; values: cint64Ptr; num_values: cint); cdecl; external;
procedure TF_SetAttrFloat(desc: TF_OperationDescriptionPtr; attr_name: charPtr; value: cfloat); cdecl; external;
procedure TF_SetAttrFloatList(desc: TF_OperationDescriptionPtr; attr_name: charPtr; values: cfloatPtr; num_values: cint); cdecl; external;
procedure TF_SetAttrBool(desc: TF_OperationDescriptionPtr; attr_name: charPtr; value: char); cdecl; external;
procedure TF_SetAttrBoolList(desc: TF_OperationDescriptionPtr; attr_name: charPtr; values: charPtr; num_values: cint); cdecl; external;
procedure TF_SetAttrType(desc: TF_OperationDescriptionPtr; attr_name: charPtr; value: TF_DataType); cdecl; external;
procedure TF_SetAttrTypeList(desc: TF_OperationDescriptionPtr; attr_name: charPtr; values: TF_DataTypePtr; num_values: cint); cdecl; external;
procedure TF_SetAttrFuncName(desc: TF_OperationDescriptionPtr; attr_name: charPtr; value: charPtr; length: csize_t); cdecl; external;
procedure TF_SetAttrShape(desc: TF_OperationDescriptionPtr; attr_name: charPtr; dims: cint64Ptr; num_dims: cint); cdecl; external;
procedure TF_SetAttrShapeList(desc: TF_OperationDescriptionPtr; attr_name: charPtr; dims: cint64PtrPtr; num_dims: cintPtr; num_shapes: cint); cdecl; external;
procedure TF_SetAttrTensorShapeProto(desc: TF_OperationDescriptionPtr; attr_name: charPtr; proto: pointer; proto_len: csize_t; status: TF_StatusPtr); cdecl; external;
procedure TF_SetAttrTensorShapeProtoList(desc: TF_OperationDescriptionPtr; attr_name: charPtr; protos: pointerPtr; proto_lens: csize_tPtr; num_shapes: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_SetAttrTensor(desc: TF_OperationDescriptionPtr; attr_name: charPtr; value: TF_TensorPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_SetAttrTensorList(desc: TF_OperationDescriptionPtr; attr_name: charPtr; values: TF_TensorPtr; num_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_SetAttrValueProto(desc: TF_OperationDescriptionPtr; attr_name: charPtr; proto: pointer; proto_len: csize_t; status: TF_StatusPtr); cdecl; external;
function TF_FinishOperation(desc: TF_OperationDescriptionPtr; status: TF_StatusPtr): TF_OperationPtr; cdecl; external;
function TF_OperationName(oper: TF_OperationPtr): charPtr; cdecl; external;
function TF_OperationOpType(oper: TF_OperationPtr): charPtr; cdecl; external;
function TF_OperationDevice(oper: TF_OperationPtr): charPtr; cdecl; external;
function TF_OperationNumOutputs(oper: TF_OperationPtr): cint; cdecl; external;
function TF_OperationOutputType(oper_out: TF_Output): TF_DataType; cdecl; external;
function TF_OperationOutputListLength(oper: TF_OperationPtr; arg_name: charPtr; status: TF_StatusPtr): cint; cdecl; external;
function TF_OperationNumInputs(oper: TF_OperationPtr): cint; cdecl; external;
function TF_OperationInputType(oper_in: TF_Input): TF_DataType; cdecl; external;
function TF_OperationInputListLength(oper: TF_OperationPtr; arg_name: charPtr; status: TF_StatusPtr): cint; cdecl; external;
function TF_OperationInput(oper_in: TF_Input): TF_Output; cdecl; external;
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
function TF_OperationGetAttrMetadata(oper: TF_OperationPtr; attr_name: charPtr; status: TF_StatusPtr): TF_AttrMetadata; cdecl; external;
procedure TF_OperationGetAttrString(oper: TF_OperationPtr; attr_name: charPtr; value: pointer; max_length: csize_t; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrStringList(oper: TF_OperationPtr; attr_name: charPtr; values: pointerPtr; lengths: csize_tPtr; max_values: cint; storage: pointer; storage_size: csize_t; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrInt(oper: TF_OperationPtr; attr_name: charPtr; value: cint64Ptr; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrIntList(oper: TF_OperationPtr; attr_name: charPtr; values: cint64Ptr; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrFloat(oper: TF_OperationPtr; attr_name: charPtr; value: cfloatPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrFloatList(oper: TF_OperationPtr; attr_name: charPtr; values: cfloatPtr; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrBool(oper: TF_OperationPtr; attr_name: charPtr; value: charPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrBoolList(oper: TF_OperationPtr; attr_name: charPtr; values: charPtr; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrType(oper: TF_OperationPtr; attr_name: charPtr; value: TF_DataTypePtr; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrTypeList(oper: TF_OperationPtr; attr_name: charPtr; values: TF_DataTypePtr; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrShape(oper: TF_OperationPtr; attr_name: charPtr; value: cint64Ptr; num_dims: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrShapeList(oper: TF_OperationPtr; attr_name: charPtr; dims: cint64PtrPtr; num_dims: cintPtr; num_shapes: cint; storage: cint64Ptr; storage_size: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrTensorShapeProto(oper: TF_OperationPtr; attr_name: charPtr; value: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrTensorShapeProtoList(oper: TF_OperationPtr; attr_name: charPtr; values: TF_BufferPtr; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrTensor(oper: TF_OperationPtr; attr_name: charPtr; value: TF_TensorPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrTensorList(oper: TF_OperationPtr; attr_name: charPtr; values: TF_TensorPtr; max_values: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_OperationGetAttrValueProto(oper: TF_OperationPtr; attr_name: charPtr; output_attr_value: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
function TF_GraphOperationByName(graph: TF_GraphPtr; oper_name: charPtr): TF_OperationPtr; cdecl; external;
function TF_GraphNextOperation(graph: TF_GraphPtr; pos: csize_tPtr): TF_OperationPtr; cdecl; external;
procedure TF_GraphToGraphDef(graph: TF_GraphPtr; output_graph_def: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_GraphGetOpDef(graph: TF_GraphPtr; op_name: charPtr; output_op_def: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_GraphVersions(graph: TF_GraphPtr; output_version_def: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;

type
  TF_ImportGraphDefOptions = OpaqueType;
  TF_ImportGraphDefOptionsPtr = ^TF_ImportGraphDefOptions;
function TF_NewImportGraphDefOptions: TF_ImportGraphDefOptionsPtr; cdecl; external;
procedure TF_DeleteImportGraphDefOptions(opts: TF_ImportGraphDefOptionsPtr); cdecl; external;
procedure TF_ImportGraphDefOptionsSetPrefix(opts: TF_ImportGraphDefOptionsPtr; prefix: charPtr); cdecl; external;
procedure TF_ImportGraphDefOptionsSetUniquifyNames(opts: TF_ImportGraphDefOptionsPtr; uniquify_names: char); cdecl; external;
procedure TF_ImportGraphDefOptionsSetUniquifyPrefix(opts: TF_ImportGraphDefOptionsPtr; uniquify_prefix: char); cdecl; external;
procedure TF_ImportGraphDefOptionsAddInputMapping(opts: TF_ImportGraphDefOptionsPtr; src_name: charPtr; src_index: cint; dst: TF_Output); cdecl; external;
procedure TF_ImportGraphDefOptionsRemapControlDependency(opts: TF_ImportGraphDefOptionsPtr; src_name: charPtr; dst: TF_OperationPtr); cdecl; external;
procedure TF_ImportGraphDefOptionsAddControlDependency(opts: TF_ImportGraphDefOptionsPtr; oper: TF_OperationPtr); cdecl; external;
procedure TF_ImportGraphDefOptionsAddReturnOutput(opts: TF_ImportGraphDefOptionsPtr; oper_name: charPtr; index: cint); cdecl; external;
function TF_ImportGraphDefOptionsNumReturnOutputs(opts: TF_ImportGraphDefOptionsPtr): cint; cdecl; external;
procedure TF_ImportGraphDefOptionsAddReturnOperation(opts: TF_ImportGraphDefOptionsPtr; oper_name: charPtr); cdecl; external;
function TF_ImportGraphDefOptionsNumReturnOperations(opts: TF_ImportGraphDefOptionsPtr): cint; cdecl; external;

type
  TF_ImportGraphDefResults = OpaqueType;
  TF_ImportGraphDefResultsPtr = ^TF_ImportGraphDefResults;
procedure TF_ImportGraphDefResultsReturnOutputs(results: TF_ImportGraphDefResultsPtr; num_outputs: cintPtr; outputs: TF_OutputPtr); cdecl; external;
procedure TF_ImportGraphDefResultsReturnOperations(results: TF_ImportGraphDefResultsPtr; num_opers: cintPtr; opers: TF_OperationPtr); cdecl; external;
procedure TF_ImportGraphDefResultsMissingUnusedInputMappings(results: TF_ImportGraphDefResultsPtr; num_missing_unused_input_mappings: cintPtr; src_names: charPtrPtrPtr; src_indexes: cintPtrPtr); cdecl; external;
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
    name: charPtr;
    end;
  TF_WhileParamsPtr = ^TF_WhileParams;
function TF_NewWhile(g: TF_GraphPtr; inputs: TF_OutputPtr; ninputs: cint; status: TF_StatusPtr): TF_WhileParams; cdecl; external;
procedure TF_FinishWhile(params: TF_WhileParamsPtr; status: TF_StatusPtr; outputs: TF_OutputPtr); cdecl; external;
procedure TF_AbortWhile(params: TF_WhileParamsPtr); cdecl; external;
procedure TF_AddGradients(g: TF_GraphPtr; y: TF_OutputPtr; ny: cint; x: TF_OutputPtr; nx: cint; dx: TF_OutputPtr; status: TF_StatusPtr; dy: TF_OutputPtr); cdecl; external;
function TF_GraphToFunction(fn_body: TF_GraphPtr; fn_name: charPtr; append_hash_to_fn_name: char; num_opers: cint; opers: TF_OperationPtr; ninputs: cint; inputs: TF_OutputPtr; noutputs: cint; outputs: TF_OutputPtr; output_names: charPtrPtr; opts: TF_FunctionOptionsPtr; description: charPtr; status: TF_StatusPtr): TF_FunctionPtr; cdecl; external;
procedure TF_FunctionToFunctionDef(func: TF_FunctionPtr; output_func_def: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
function TF_FunctionImportFunctionDef(proto: pointer; proto_len: csize_t; status: TF_StatusPtr): TF_FunctionPtr; cdecl; external;
procedure TF_FunctionSetAttrValueProto(func: TF_FunctionPtr; attr_name: charPtr; proto: pointer; proto_len: csize_t; status: TF_StatusPtr); cdecl; external;
procedure TF_FunctionGetAttrValueProto(func: TF_FunctionPtr; attr_name: charPtr; output_attr_value: TF_BufferPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_DeleteFunction(func: TF_FunctionPtr); cdecl; external;
function TF_TryEvaluateConstant(graph: TF_GraphPtr; output: TF_Output; result_: TF_TensorPtr; status: TF_StatusPtr): char; cdecl; external;

type
  TF_Session = OpaqueType;
  TF_SessionPtr = ^TF_Session;
function TF_NewSession(graph: TF_GraphPtr; opts: TF_SessionOptionsPtr; status: TF_StatusPtr): TF_SessionPtr; cdecl; external;
function TF_LoadSessionFromSavedModel(session_options: TF_SessionOptionsPtr; run_options: TF_BufferPtr; export_dir: charPtr; tags: charPtrPtr; tags_len: cint; graph: TF_GraphPtr; meta_graph_def: TF_BufferPtr; status: TF_StatusPtr): TF_SessionPtr; cdecl; external;
procedure TF_CloseSession(param1: TF_SessionPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_DeleteSession(param1: TF_SessionPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_SessionRun(session: TF_SessionPtr; run_options: TF_BufferPtr; inputs: TF_OutputPtr; input_values: TF_TensorPtr; ninputs: cint; outputs: TF_OutputPtr; output_values: TF_TensorPtr; noutputs: cint; target_opers: TF_OperationPtr; ntargets: cint; run_metadata: TF_BufferPtr; param12: TF_StatusPtr); cdecl; external;
procedure TF_SessionPRunSetup(param1: TF_SessionPtr; inputs: TF_OutputPtr; ninputs: cint; outputs: TF_OutputPtr; noutputs: cint; target_opers: TF_OperationPtr; ntargets: cint; handle: charPtrPtr; param9: TF_StatusPtr); cdecl; external;
procedure TF_SessionPRun(param1: TF_SessionPtr; handle: charPtr; inputs: TF_OutputPtr; input_values: TF_TensorPtr; ninputs: cint; outputs: TF_OutputPtr; output_values: TF_TensorPtr; noutputs: cint; target_opers: TF_OperationPtr; ntargets: cint; param11: TF_StatusPtr); cdecl; external;
procedure TF_DeletePRunHandle(handle: charPtr); cdecl; external;

type
  TF_DeprecatedSession = OpaqueType;
  TF_DeprecatedSessionPtr = ^TF_DeprecatedSession;
function TF_NewDeprecatedSession(param1: TF_SessionOptionsPtr; status: TF_StatusPtr): TF_DeprecatedSessionPtr; cdecl; external;
procedure TF_CloseDeprecatedSession(param1: TF_DeprecatedSessionPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_DeleteDeprecatedSession(param1: TF_DeprecatedSessionPtr; status: TF_StatusPtr); cdecl; external;
procedure TF_Reset(opt: TF_SessionOptionsPtr; containers: charPtrPtr; ncontainers: cint; status: TF_StatusPtr); cdecl; external;
procedure TF_ExtendGraph(param1: TF_DeprecatedSessionPtr; proto: pointer; proto_len: csize_t; param4: TF_StatusPtr); cdecl; external;
procedure TF_Run(param1: TF_DeprecatedSessionPtr; run_options: TF_BufferPtr; input_names: charPtrPtr; inputs: TF_TensorPtr; ninputs: cint; output_names: charPtrPtr; outputs: TF_TensorPtr; noutputs: cint; target_oper_names: charPtrPtr; ntargets: cint; run_metadata: TF_BufferPtr; param12: TF_StatusPtr); cdecl; external;
procedure TF_PRunSetup(param1: TF_DeprecatedSessionPtr; input_names: charPtrPtr; ninputs: cint; output_names: charPtrPtr; noutputs: cint; target_oper_names: charPtrPtr; ntargets: cint; handle: charPtrPtr; param9: TF_StatusPtr); cdecl; external;
procedure TF_PRun(param1: TF_DeprecatedSessionPtr; handle: charPtr; input_names: charPtrPtr; inputs: TF_TensorPtr; ninputs: cint; output_names: charPtrPtr; outputs: TF_TensorPtr; noutputs: cint; target_oper_names: charPtrPtr; ntargets: cint; param11: TF_StatusPtr); cdecl; external;

type
  TF_DeviceList = OpaqueType;
  TF_DeviceListPtr = ^TF_DeviceList;
function TF_SessionListDevices(session: TF_SessionPtr; status: TF_StatusPtr): TF_DeviceListPtr; cdecl; external;
function TF_DeprecatedSessionListDevices(session: TF_DeprecatedSessionPtr; status: TF_StatusPtr): TF_DeviceListPtr; cdecl; external;
procedure TF_DeleteDeviceList(list: TF_DeviceListPtr); cdecl; external;
function TF_DeviceListCount(list: TF_DeviceListPtr): cint; cdecl; external;
function TF_DeviceListName(list: TF_DeviceListPtr; index: cint; status: TF_StatusPtr): charPtr; cdecl; external;
function TF_DeviceListType(list: TF_DeviceListPtr; index: cint; status: TF_StatusPtr): charPtr; cdecl; external;
function TF_DeviceListMemoryBytes(list: TF_DeviceListPtr; index: cint; status: TF_StatusPtr): cint64; cdecl; external;

type
  TF_Library = OpaqueType;
  TF_LibraryPtr = ^TF_Library;
function TF_LoadLibrary(library_filename: charPtr; status: TF_StatusPtr): TF_LibraryPtr; cdecl; external;
function TF_GetOpList(lib_handle: TF_LibraryPtr): TF_Buffer; cdecl; external;
procedure TF_DeleteLibraryHandle(lib_handle: TF_LibraryPtr); cdecl; external;
function TF_GetAllOpList: TF_BufferPtr; cdecl; external;

type
  TF_ApiDefMap = OpaqueType;
  TF_ApiDefMapPtr = ^TF_ApiDefMap;
function TF_NewApiDefMap(op_list_buffer: TF_BufferPtr; status: TF_StatusPtr): TF_ApiDefMapPtr; cdecl; external;
procedure TF_DeleteApiDefMap(apimap: TF_ApiDefMapPtr); cdecl; external;
procedure TF_ApiDefMapPut(api_def_map: TF_ApiDefMapPtr; text: charPtr; text_len: csize_t; status: TF_StatusPtr); cdecl; external;
function TF_ApiDefMapGet(api_def_map: TF_ApiDefMapPtr; name: charPtr; name_len: csize_t; status: TF_StatusPtr): TF_BufferPtr; cdecl; external;

type
  TF_Shape=array of Int64;
  TF_ShapePtr=^TF_Shape;
  TF_FloatList=array of cfloat;
  TF_FuncnameList=array of string;
  TF_IntList=array of cint64;
  TF_ShapeList=array of TF_Shape;
  TF_StringList=array of string;
  TF_TypeList=array of TF_DataType;


implementation

uses
  SysUtils;

function TF_CheckStatus(AStatusPtr:TF_StatusPtr):Boolean;
  begin
  result:=TF_GetCode(AStatusPtr)=TF_OK;
  if not result then
    raise exception.create(TF_Message(AStatusPtr));
  end;

end.

