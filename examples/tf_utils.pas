unit tf_utils;

//**********************************************************************************************************************************
//
//  Pascal methods to make use and debugging of Tensorflow in Pascal easier
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
//    17/01/2023 Unimportant compiler warnings are "fixed"
//               Unimportant compiler hints suppressed
//    18/01/2023 PrintTensorData made available for TF_STRING type tensors as well
//
//**********************************************************************************************************************************
//
//  Description
//
//  This is a unit strictly for experimental use. The content of it is not considered stable, it can change any time without even
//  a change log manintained. If you can use something from it, it is good news, but if you cannot, no reason to complain.
//
//**********************************************************************************************************************************

interface

uses
  SysUtils,                            // Needed for exception handling and string conversions
  tf_api,                              // The pascal version of c_api.h
  tf_tensors,                          // The Tensor handling
  tf_wrapper;                          // TGraphExt and the Specific Exec<oper> functions

procedure PrintTensorShape(const ATensor:TF_TensorPtr; const AName:string='');
procedure PrintTensorData(const ATensor:TF_TensorPtr; const AName:string='');
function CreateTensorSingleRandom(const AShape:array of Int64; const MinValue:single; const MaxValue:single):TF_TensorPtr;

implementation

procedure PrintTensorShape(const ATensor:TF_TensorPtr; const AName:string='');
  var
    Shape:TF_Shape;
    DataType:TF_DataType;
    I:Integer;
  begin
  Write('Tensor details ',AName,': Type: ');
  DataType:=TF_TensorType(ATensor);
  case DataType of
    TF_BOOL:      write('Boolean');
    TF_INT8:      write('Int8');
    TF_INT16:     write('Int16');
    TF_INT32:     write('Int32');
    TF_INT64:     write('Int64');
    TF_UINT8:     write('UInt8');
    TF_UINT16:    write('UInt16');
    TF_UINT32:    write('UInt32');
    TF_UINT64:    write('UInt64');
    TF_HALF:      write('Half');
    TF_FLOAT:     write('Float');
    TF_DOUBLE:    write('Double');
    TF_COMPLEX64: write('Complex64');
    TF_COMPLEX128:write('Complex128');
    TF_STRING:    write('String');
    TF_VARIANT:   write('Variant');
    else       write(DataType);
    end;
  write('. Dimensions: ');
  Shape:=GetTensorShape(ATensor);
  for I:=0 to length(Shape)-1 do
    begin
    write(Shape[i]);
    if I<Length(Shape)-1 then
      write('x');
    end;
  write(' Total length:',TF_TensorByteSize(ATensor));
  writeln;
  end;
procedure PrintTensorData(const ATensor:TF_TensorPtr; const AName:string='');
  // to simplify life, all tensors are printed as Single (otherwise the Variant type would need to be split as per its type.
  const
    AFormat:string='%10.5f';
  var
    i,j:integer;
    Index:TF_Shape;
    DataType:TF_DataType;
  begin
  if AName<>'' then writeln(AName);

  DataType:=TF_TensorType(ATensor);
  if DataType = TF_String then
    begin
    for i:= 0 to GetTensorScalarCount(ATensor)-1 do
      begin
      Index:=GetTensorScalarToIndex(ATensor,i);
      for j:=0 to Length(Index)-1 do
        write(Index[j],' ');
      writeln('"',GetTensorValue(ATensor,Index),'"');
      end;
    exit;
    end;

  case Length(GetTensorShape(ATensor)) of
    0:writeln(Format(AFormat,[Single(GetTensorValue(ATensor,[])){%H-}]));
    1:for i:=0 to GetTensorScalarCount(ATensor)-1 do
        writeln(Format(AFormat,[Single(GetTensorValue(ATensor,[i])){%H-}]));
    2:begin
      for i:=0 to GetTensorShape(ATensor)[0]-1 do
        begin
        for j:=0 to GetTensorShape(ATensor)[1]-1 do
          write(Format(AFormat,[Single(GetTensorValue(ATensor,[i,j])){%H-}]),chr(9));
        writeln;
        end;
      end
    else
      begin
      for i:=0 to GetTensorScalarCount(ATensor)-1 do
        begin
        Index:=GetTensorScalarToIndex(ATensor,i);
        for j:=0 to Length(Index)-1 do
          write(Index[j],' ');
        writeln('    ',Format(AFormat,[Single(GetTensorValue(ATensor,Index)){%H-}]));
        end;
      end;
    end;
  end;

function CreateTensorSingleRandom(const AShape:array of Int64; const MinValue:single; const MaxValue:single):TF_TensorPtr;
  begin
  result:=ExecRandomUniform(CreateTensorInt64([Length(AShape)],AShape),random(MaxInt),random(MaxInt),TF_FLOAT,true);
  result:=ExecMul(result,CreateTensorSingle(MaxValue-MinValue),true,true);
  result:=ExecAdd(result,CreateTensorSingle(MinValue),true,true);
  end;

end.

