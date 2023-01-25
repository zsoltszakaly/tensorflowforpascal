unit tf_tensors;

//**********************************************************************************************************************************
//
//  Pascal methods to create and manage TensorFlow based Tensors
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
//  Change log:
//    13/02/2020 Initial version
//    15/02/2020 CreateTensorString added
//    18/02/2020 CreateTensorString for multiple strings added
//    22/10/2021 More CreateTensor versions are created for integer types as well
//    17/01/2023 Update to reflect TF change from 1.15 to 2.11 in CreateTensorString
//               Unimportant compiler hints suppressed
//    18/01/2023 GetTensorValue extended to TF_STRING
//    19/01/2023 Added GetTensorDataTypeSize
//               Fixed the CreateTensorString routines
//    22/01/2023 CreateTensorBool added
//
//**********************************************************************************************************************************
//
//  Description
//
//  The Tensor operations has three parts
//    General administration
//    Creation of various Tensor types
//    Data move to and from a Tensor
//
//**********************************************************************************************************************************

interface

uses
  SysUtils,                            // Needed for exception handling and string conversions
  Variants,                            // Needed for the Variant type used to read individual data from a Tensor
  tf_api;                              // The pascal version of c_api.h

//**********************************************************************************************************************************
//  General administration
//**********************************************************************************************************************************

function GetTensorShape(const ATensor:TF_TensorPtr):TF_Shape;
function GetTensorScalarCount(const ATensor:TF_TensorPtr):Int64;
function GetTensorIndexToScalar(const ATensor:TF_TensorPtr; const AIndex:array of Int64):Int64;
function GetTensorScalarToIndex(const ATensor:TF_TensorPtr; const AScalar:Int64):TF_Shape;
function GetTensorDataTypeSize(const aTensor: TF_TensorPtr): Int64;

//**********************************************************************************************************************************
//  Creation of various Tensor types
//**********************************************************************************************************************************

// the generic function to create a Tensor without adding data to it
function CreateTensor(ADataType:TF_DataType; const AShape:array of Int64):TF_TensorPtr;

// there are four versions: (a) constant, (b) vector, (c) shape and an appropriate array, (d) shape, pointer and length
function CreateTensorBool(AData:boolean):TF_TensorPtr;
function CreateTensorBool(const AData:array of boolean):TF_TensorPtr;
function CreateTensorBool(const AShape:array of Int64; const AData:array of boolean):TF_TensorPtr;
function CreateTensorBool(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
function CreateTensorInt8(AData:Int8):TF_TensorPtr;
function CreateTensorInt8(const AData:array of Int8):TF_TensorPtr;
function CreateTensorInt8(const AShape:array of Int64; const AData:array of Int8):TF_TensorPtr;
function CreateTensorInt8(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
function CreateTensorInt16(AData:Int16):TF_TensorPtr;
function CreateTensorInt16(const AData:array of Int16):TF_TensorPtr;
function CreateTensorInt16(const AShape:array of Int64; const AData:array of Int16):TF_TensorPtr;
function CreateTensorInt16(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
function CreateTensorInt32(AData:Int32):TF_TensorPtr;
function CreateTensorInt32(const AData:array of Int32):TF_TensorPtr;
function CreateTensorInt32(const AShape:array of Int64; const AData:array of Int32):TF_TensorPtr;
function CreateTensorInt32(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
function CreateTensorInt64(AData:Int64):TF_TensorPtr;
function CreateTensorInt64(const AData:array of Int64):TF_TensorPtr;
function CreateTensorInt64(const AShape:array of Int64; const AData:array of Int64):TF_TensorPtr;
function CreateTensorInt64(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
function CreateTensorUInt8(AData:UInt8):TF_TensorPtr;
function CreateTensorUInt8(const AData:array of UInt8):TF_TensorPtr;
function CreateTensorUInt8(const AShape:array of Int64; const AData:array of UInt8):TF_TensorPtr;
function CreateTensorUInt8(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
function CreateTensorUInt16(AData:UInt16):TF_TensorPtr;
function CreateTensorUInt16(const AData:array of UInt16):TF_TensorPtr;
function CreateTensorUInt16(const AShape:array of Int64; const AData:array of UInt16):TF_TensorPtr;
function CreateTensorUInt16(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
function CreateTensorUInt32(AData:UInt32):TF_TensorPtr;
function CreateTensorUInt32(const AData:array of UInt32):TF_TensorPtr;
function CreateTensorUInt32(const AShape:array of Int64; const AData:array of UInt32):TF_TensorPtr;
function CreateTensorUInt32(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
function CreateTensorUInt64(AData:UInt64):TF_TensorPtr;
function CreateTensorUInt64(const AData:array of UInt64):TF_TensorPtr;
function CreateTensorUInt64(const AShape:array of Int64; const AData:array of UInt64):TF_TensorPtr;
function CreateTensorUInt64(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
function CreateTensorSingle(const AData:Single):TF_TensorPtr;
function CreateTensorSingle(const AData:array of Single):TF_TensorPtr;
function CreateTensorSingle(const AShape:array of Int64; const AData:array of Single):TF_TensorPtr;
function CreateTensorSingle(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;

function CreateTensorString(const AData:PChar):TF_TensorPtr;
function CreateTensorString(const AShape:array of Int64; const AData:array of PChar):TF_TensorPtr;

// other tpyes, e.g. Complex can be added later, but the generic one can always be used

//**********************************************************************************************************************************
//  Data move to and from a Tensor
//**********************************************************************************************************************************

procedure WriteTensorData(const ATensor:TF_TensorPtr; const AData; ADataLength:Int64);
procedure ReadTensorData(const ATensor:TF_TensorPtr; const AData; ADataLength:Int64);
function GetTensorValue(const ATensor:TF_TensorPtr; const AIndex:array of Int64):Variant;

implementation

//**********************************************************************************************************************************
//  General administration
//**********************************************************************************************************************************

function GetTensorShape(const ATensor:TF_TensorPtr):TF_Shape;
  var I:integer;
  begin
  result:=nil;
  SetLength(result,TF_NumDims(ATensor));
  for I:=0 to length(result)-1 do
    result[i]:=TF_Dim(ATensor,I);
  end;
function GetTensorScalarCount(const ATensor:TF_TensorPtr):Int64;
  var
    TensorShape : TF_Shape;
    i : integer;
  begin
  TensorShape := GetTensorShape(ATensor);
  result := 1;
  for i := 0 to length(TensorShape) - 1 do
    result := result * TF_Dim(ATensor, i);
  end;
function GetTensorIndexToScalar(const ATensor:TF_TensorPtr; const AIndex:array of Int64):Int64;
  var
    Shape:TF_Shape;
    Index:integer;
    Scalar:Int64;
  begin
  Shape:=GetTensorShape(ATensor);
  result:=0;
  Scalar:=1;
  if Length(Shape)<>Length(AIndex) then
    exit;
  for Index:=Length(Shape)-1 downto 0 do
    begin
    result:=result+Scalar*AIndex[Index];
    Scalar:=Scalar*Shape[Index];
    end;
  end;
function GetTensorScalarToIndex(const ATensor:TF_TensorPtr; const AScalar:Int64):TF_Shape;
  var
    Shape:TF_Shape;
    Index:integer;
    Scalar:Int64;
  begin
  result:=nil;
  Shape:=GetTensorShape(ATensor);
  SetLength(result,Length(Shape));
  Scalar:=AScalar;
  for Index:=Length(Shape)-1 downto 0 do
    begin
    result[Index]:=Scalar mod Shape[Index];
    Scalar:=Scalar div Shape[Index];
    end;
  end;
function GetTensorDataTypeSize(const aTensor: TF_TensorPtr): Int64;
  begin
  result := TF_DataTypeSize(TF_TensorType(ATensor));
  if result = 0 then // try to figure it out, especially for TF_String type
    result := TF_TensorByteSize(ATensor) div GetTensorScalarCount(aTensor);
  end;

//**********************************************************************************************************************************
//  Creation of various Tensor types
//**********************************************************************************************************************************

function CreateTensor(ADataType:TF_DataType; const AShape:array of Int64):TF_TensorPtr;
  var
    DataLength:QWord;
    TensorPtr:TF_TensorPtr;
    I:Integer;
  begin
  DataLength:=TF_DataTypeSize(ADataType);
  for I:=0 to Length(AShape)-1 do
    DataLength:=DataLength*AShape[I];
  TensorPtr:=TF_AllocateTensor(ADataType, @AShape[0], Length(AShape), DataLength);
  if not Assigned(TensorPtr) then
    raise Exception.Create('Tensor cannot be created');
  result:=TensorPtr;
  end;
function CreateTensorBool(AData:boolean):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_BOOL,[]);
  WriteTensorData(result, AData, SizeOf(AData));
  end;
function CreateTensorBool(const AData:array of boolean):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_BOOL,[length(AData)]);
  WriteTensorData(result, AData[0], Length(AData)*SizeOf(boolean));
  end;
function CreateTensorBool(const AShape:array of Int64; const AData:array of boolean):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_BOOL,AShape);
  WriteTensorData(result, AData[0], Length(AData)*SizeOf(boolean));
  end;
function CreateTensorBool(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_BOOL,AShape);
  WriteTensorData(result,AData,ALength);
  end;
function CreateTensorInt8(AData:Int8):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT8,[]);
  WriteTensorData(result, AData, SizeOf(AData));
  end;
function CreateTensorInt8(const AData:array of Int8):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT8,[length(AData)]);
  WriteTensorData(result, AData[0], Length(AData)*SizeOf(Int8));
  end;
function CreateTensorInt8(const AShape:array of Int64; const AData:array of Int8):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT8,AShape);
  WriteTensorData(result, AData[0], Length(AData)*SizeOf(Int8));
  end;
function CreateTensorInt8(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT8,AShape);
  WriteTensorData(result,AData,ALength);
  end;
function CreateTensorInt16(AData:Int16):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT16,[]);
  WriteTensorData(result, AData, SizeOf(AData));
  end;
function CreateTensorInt16(const AData:array of Int16):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT16,[length(AData)]);
  WriteTensorData(result, AData[0], Length(AData)*SizeOf(Int16));
  end;
function CreateTensorInt16(const AShape:array of Int64; const AData:array of Int16):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT16,AShape);
  WriteTensorData(result, AData[0], Length(AData)*SizeOf(Int16));
  end;
function CreateTensorInt16(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT16,AShape);
  WriteTensorData(result,AData,ALength);
  end;
function CreateTensorInt32(AData:Int32):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT32,[]);
  WriteTensorData(result, AData, SizeOf(AData));
  end;
function CreateTensorInt32(const AData:array of Int32):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT32,[length(AData)]);
  WriteTensorData(result,AData[0],Length(AData)*SizeOf(Int32));
  end;
function CreateTensorInt32(const AShape:array of Int64; const AData:array of Int32):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT32,AShape);
  WriteTensorData(result,AData[0],Length(AData)*SizeOf(Int32));
  end;
function CreateTensorInt32(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT32,AShape);
  WriteTensorData(result,AData,ALength);
  end;
function CreateTensorInt64(AData:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT64,[]);
  WriteTensorData(result, AData, SizeOf(AData));
  end;
function CreateTensorInt64(const AData:array of Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT64,[length(AData)]);
  WriteTensorData(result,AData[0],Length(AData)*SizeOf(Int64));
  end;
function CreateTensorInt64(const AShape:array of Int64; const AData:array of Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT64,AShape);
  WriteTensorData(result,AData[0],Length(AData)*SizeOf(Int64));
  end;
function CreateTensorInt64(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_INT64,AShape);
  WriteTensorData(result,AData,ALength);
  end;
function CreateTensorUInt8(AData:UInt8):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UINT8,[]);
  WriteTensorData(result, AData, SizeOf(AData));
  end;
function CreateTensorUInt8(const AData:array of Byte):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UINT8,[length(AData)]);
  WriteTensorData(result, AData[0], Length(AData)*SizeOf(Byte));
  end;
function CreateTensorUInt8(const AShape:array of Int64; const AData:array of Byte):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UINT8,AShape);
  WriteTensorData(result, AData[0], Length(AData)*SizeOf(Byte));
  end;
function CreateTensorUInt8(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UINT8,AShape);
  WriteTensorData(result,AData,ALength);
  end;
function CreateTensorUInt16(AData:UInt16):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt16,[]);
  WriteTensorData(result, AData, SizeOf(AData));
  end;
function CreateTensorUInt16(const AData:array of UInt16):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt16,[length(AData)]);
  WriteTensorData(result, AData[0], Length(AData)*SizeOf(UInt16));
  end;
function CreateTensorUInt16(const AShape:array of Int64; const AData:array of UInt16):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt16,AShape);
  WriteTensorData(result, AData[0], Length(AData)*SizeOf(UInt16));
  end;
function CreateTensorUInt16(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt16,AShape);
  WriteTensorData(result,AData,ALength);
  end;
function CreateTensorUInt32(AData:UInt32):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt32,[]);
  WriteTensorData(result, AData, SizeOf(AData));
  end;
function CreateTensorUInt32(const AData:array of UInt32):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt32,[length(AData)]);
  WriteTensorData(result,AData[0],Length(AData)*SizeOf(UInt32));
  end;
function CreateTensorUInt32(const AShape:array of Int64; const AData:array of UInt32):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt32,AShape);
  WriteTensorData(result,AData[0],Length(AData)*SizeOf(UInt32));
  end;
function CreateTensorUInt32(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt32,AShape);
  WriteTensorData(result,AData,ALength);
  end;
function CreateTensorUInt64(AData:UInt64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt64,[]);
  WriteTensorData(result, AData, SizeOf(AData));
  end;
function CreateTensorUInt64(const AData:array of UInt64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt64,[length(AData)]);
  WriteTensorData(result,AData[0],Length(AData)*SizeOf(UInt64));
  end;
function CreateTensorUInt64(const AShape:array of Int64; const AData:array of UInt64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt64,AShape);
  WriteTensorData(result,AData[0],Length(AData)*SizeOf(UInt64));
  end;
function CreateTensorUInt64(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_UInt64,AShape);
  WriteTensorData(result,AData,ALength);
  end;
function CreateTensorSingle(const AData:Single):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_FLOAT,[]);
  WriteTensorData(result, AData, SizeOf(Single));
  end;
function CreateTensorSingle(const AData:array of Single):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_FLOAT,[Length(AData)]);
  WriteTensorData(result,AData[0],Length(AData)*SizeOf(Single));
  end;
function CreateTensorSingle(const AShape:array of Int64; const AData:array of Single):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_FLOAT,AShape);
  WriteTensorData(result,AData[0],Length(AData)*SizeOf(Single));
  end;
function CreateTensorSingle(const AShape:array of Int64; const AData; ALength:Int64):TF_TensorPtr;
  begin
  result:=CreateTensor(TF_FLOAT,AShape);
  WriteTensorData(result,AData,ALength);
  end;
function CreateTensorString(const AData:PChar):TF_TensorPtr;
  var
    Status:TF_StatusPtr;
    TString:TF_TString;
  begin
  Status:=TF_NewStatus;
  // 18.01.2023 New version to reflect changes in TF API
  result:=TF_AllocateTensor(TF_String,nil,0,sizeof(TString));
  TF_StringInit(@TString);
  TF_StringCopy(@TString, AData, strlen(AData));
  WriteTensorData(result,TString,sizeof(TString));
  (* Previous version that worked under 1.15
  result:=TF_AllocateTensor(TF_STRING,nil,0,8+TF_StringEncodedSize(strlen(AData)));
  TF_StringEncode(AData,strlen(AData),TF_TensorData(result)+8,TF_StringEncodedSize(strlen(AData)),Status);
  *)
  TF_CheckStatus(Status);
  end;
function CreateTensorString(const AShape:array of Int64; const AData:array of PChar):TF_TensorPtr;
  var
    Status:TF_StatusPtr;
    TString:TF_TString;
//    TotalLength:Int64;
    i:integer;
  begin
  // No check to make sure that it has the same number of strings required by the Shape
  (* Previous version
  Status:=TF_NewStatus;
  TotalLength:=0;
  for i:=0 to Length(AData)-1 do
    TotalLength:=TotalLength+8+TF_StringEncodedSize(strlen(AData[i]));
  result:=TF_AllocateTensor(TF_STRING,@AShape[0],Length(AShape),TotalLength);
  TotalLength:=0;
  for i:=0 to Length(AData)-1 do
    begin
    Int64((TF_TensorData(result)+i*8)^):=TotalLength;
    TotalLength:=TotalLength+TF_StringEncodedSize(strlen(AData[i]));
    end;
  TotalLength:=8*Length(AData);
  for i:=0 to Length(AData)-1 do
    begin
    TF_StringEncode(AData[i],strlen(AData[i]),TF_TensorData(result)+TotalLength,TF_StringEncodedSize(strlen(AData[i])),Status);
    TF_CheckStatus(Status);
    TotalLength:=TotalLength+TF_StringEncodedSize(strlen(AData[i]));
    end;
  TF_CheckStatus(Status);
  *)
  // 18/01/2023 New version
  Status:=TF_NewStatus;
  result:=TF_AllocateTensor(TF_STRING,@AShape[0],Length(AShape),Length(AData) * sizeof(TF_TString));
  for i:=0 to Length(AData)-1 do
    begin
    TF_StringInit(@TString);
    TF_StringCopy(@TString, AData[i], strlen(AData[i]));
    Move(TString, (TF_TensorData(result)+i * sizeof(TString))^, sizeof(TString));
    end;
  TF_CheckStatus(Status);
  end;

//**********************************************************************************************************************************
//  Data move to and from a Tensor
//**********************************************************************************************************************************

procedure WriteTensorData(const ATensor:TF_TensorPtr; const AData; ADataLength:Int64);
  begin
  if TF_TensorByteSize(ATensor)<>ADataLength then
    begin
    if ADataLength>TF_TensorByteSize(ATensor) then
      begin
      writeln('More data provided than the size of the tensor. Not all data used.');
      ADataLength:=TF_TensorByteSize(ATensor);
      end;
    if ADataLength<TF_TensorByteSize(ATensor) then
      writeln('Less data provided than the size of the tensor. Tensor is partially filled.');
    end;
  try
    Move(AData, TF_TensorData(ATensor)^, ADataLength);
  except
    raise exception.create('WriteTensorData failed');
    end;
  end;
procedure ReadTensorData(const ATensor:TF_TensorPtr; const AData; ADataLength:Int64);
  begin
  if TF_TensorByteSize(ATensor)<>ADataLength then
    begin
    if TF_TensorByteSize(ATensor)<ADataLength then
      begin
      writeln('More data requested than the size of the tensor. Request partially filled.');
      ADataLength:=TF_TensorByteSize(ATensor);
      end;
    if TF_TensorByteSize(ATensor)>ADataLength then
      writeln('Less data requested than the size of the tensor.');
    end;
  try
    Move(TF_TensorData(ATensor)^, (@AData)^, ADataLength);
  except
    raise exception.create('ReadTensorData failed');
    end;
  end;
function GetTensorValue(const ATensor:TF_TensorPtr; const AIndex:array of Int64):Variant;
  begin
  case TF_TensorType(ATensor) of
    TF_INT8:result:={%H-}Int8((TF_TensorData(ATensor)+
                               GetTensorIndexToScalar(ATensor,AIndex)*TF_DataTypeSize(TF_TensorType(ATensor)))^);
    TF_INT16:result:={%H-}Int16((TF_TensorData(ATensor)+
                                 GetTensorIndexToScalar(ATensor,AIndex)*TF_DataTypeSize(TF_TensorType(ATensor)))^);
    TF_INT32:result:={%H-}Int32((TF_TensorData(ATensor)+
                                 GetTensorIndexToScalar(ATensor,AIndex)*TF_DataTypeSize(TF_TensorType(ATensor)))^);
    TF_INT64:result:={%H-}Int64((TF_TensorData(ATensor)+
                                 GetTensorIndexToScalar(ATensor,AIndex)*TF_DataTypeSize(TF_TensorType(ATensor)))^);
    TF_UINT8:result:={%H-}UInt8((TF_TensorData(ATensor)+
                                 GetTensorIndexToScalar(ATensor,AIndex)*TF_DataTypeSize(TF_TensorType(ATensor)))^);
    TF_UINT16:result:={%H-}UInt16((TF_TensorData(ATensor)+
                                   GetTensorIndexToScalar(ATensor,AIndex)*TF_DataTypeSize(TF_TensorType(ATensor)))^);
    TF_UINT32:result:={%H-}UInt32((TF_TensorData(ATensor)+
                                   GetTensorIndexToScalar(ATensor,AIndex)*TF_DataTypeSize(TF_TensorType(ATensor)))^);
    TF_UINT64:result:={%H-}UInt64((TF_TensorData(ATensor)+
                                   GetTensorIndexToScalar(ATensor,AIndex)*TF_DataTypeSize(TF_TensorType(ATensor)))^);
    TF_FLOAT:result:={%H-}Single((TF_TensorData(ATensor)+
                                  GetTensorIndexToScalar(ATensor,AIndex)*TF_DataTypeSize(TF_TensorType(ATensor)))^);
    TF_DOUBLE:result:={%H-}Double((TF_TensorData(ATensor)+
                                   GetTensorIndexToScalar(ATensor,AIndex)*TF_DataTypeSize(TF_TensorType(ATensor)))^);
    TF_STRING:result:={%H-}TF_StringGetDataPointer((TF_TensorData(ATensor)+
                                                   GetTensorIndexToScalar(ATensor,AIndex)*GetTensorDataTypeSize(ATensor)));
    end;
  end;

end.
