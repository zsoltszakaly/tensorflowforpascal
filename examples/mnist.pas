program mnist;

//**********************************************************************************************************************************
//
//  An MNIST implementation
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
//  Change log: 23/02/2020 Initial version
//
//**********************************************************************************************************************************
//
//  Description
//
//  This program implements an MNIST program.
//  The model is a simple 784 - 100Sigmoid - 30Sigmoid - 10 3 layer ANN.
//  The cost function is  - Label * Log(Prediction) (Cross entropy)
//  The size of the layers can easily be changed, changing some numbers in Createmodel, all other changes (Number of layers,
//    Activation function, etc.) would require the program to be changed.
//  The source of the data you need to download separately (just google for the location)
//  Normally it uses the whole training and testing set, but in LoadInputAndLabel, it is easy to change
//  The learning rate is fine tuned to maximise the improvement rate, not to maximise the learning rate. This is a bit complicated
//    but works.
//  The generated result is checked both using the cost function and creating a histogram of the prediction given for the label.
//  The preparation (initial model) and final analysis of the data is done eagerly (Exec<oper> functions), but the actual work is
//    done in a Graph using Add<oper> instructions. This is many times (my test showed 6 times) faster than doing the same with
//    Eager operations only.
//
//  It is important to note, that this is not to show "the best" or even a "good enough" model, the sole purpose is to show how the
//    Pascal interfaces can be used for a real problem. Some of the functions could be simplified (e.g. the cost function could use
//    AddXLogY instead of the two steps used), but the purpose was to show different approaches, not to optimise the program.
//  Also the loop is done outside of the Graph. Some consideration was given to put the LearningRate adjustment inside the Graph
//    too, but since the calculation is done using Scalar Single, it would not yield much benefit, but would make the Graph too
//    complicated. Also to arrange a loop using Enter-Merge-Switch-Exit-nextIteration, I could not solve (yet). If you have a
//    solution, please let me know.
//
//**********************************************************************************************************************************

uses
  SysUtils,
  tf_api,
  tf_tensors,
  tf_operations,
  tf_wrapper,
  tf_utils;

const
  LearnInputFileName : string = 'train-images-idx3-ubyte'; //Load these files from the web
  LearnLabelFileName : string = 'train-labels-idx1-ubyte';
  TestInputFileName  : string = 't10k-images-idx3-ubyte';
  TestLabelFileName  : string = 't10k-labels-idx1-ubyte';
  ModelFileName      : string = 'mnist.ann'; // This is where the result is saved

const
  cSigmoidSensitiveRange      : single = 1.31695789692482; // ln( 2 +/- sqrt(3) ), i.e. the most sensitive range of a sigmoid function
  cTanHSensitiveRange         : single = 0.65847894846241; // 0.5*ln(2+sqrt(3))
  cReLUAverageValue           : single = 3.0; // Half of ReLU6, so should be OK
  cReLuStandardDeviation      : single = 1.0; // 2-4 to be the most active range. With extra sqrt(2) from B, it is 1.5-4.5
  cInitialLearningRate                 = 1.0;
  cLearningRateMaximum        : single = 2.0; // No special reason why, can be changed
  cLearningRateReviewFrequency: integer= 50;  // Print and Save progress frequency
  cLearningRateDecreaseRate   : Single = 0.94; // In case of too big change, it is cut back to this ratio
  cLearningRateAdjustRate     : Single = 0.01; // In case of OK, the LearningRate is fine-tuned with +/- this rate

type
  TNeuronActivationFunction=(nafNone,nafBinaryStep,nafLinear,nafSigmoid,nafTanh,nafReLU,nafLeakyReLU,nafSoftmax,nafSwish);
  TNeuronLayer=
    record
      W:TF_TensorPtr;
      B:TF_TensorPtr;
      AF:TNeuronActivationFunction;
    end;
  TNeuronLayers=array of TNeuronLayer;

var
  NeuronLayers:TNeuronLayers;
  InputT:TF_TensorPtr;
  LabelT:TF_TensorPtr;
  LearningRate : Single;
  G:TGraphExt;
  S:TSession;

procedure LoadInputAndLabel(const InputFileName:string;const LabelFileName:string);
  var
    LabelFile:file;
    InputFile:file;
    RowCount:int64;
    ColCount:Int64;
    LoadBytes:array of byte;
    LoadSingles:array of Single;
    i:integer;
  begin
  AssignFile(LabelFile,LabelFileName);
  Reset(LabelFile,1);
  Seek(LabelFile,8);  // Skip first 8 bytes in file -- see file format
  RowCount:=(FileSize(LabelFile)-8);

//  RowCount:=5000; // If you want to work with less data point, here it is easy to change manually

  SetLength(LoadBytes,RowCount);
  BlockRead(LabelFile, LoadBytes[0],RowCount);
  LabelT:=ExecOneHot(CreateTensorUInt8([length(LoadBytes)],LoadBytes),
                     CreateTensorInt32(10),
                     CreateTensorSingle(1.0),
                     CreateTensorSingle(0.0),
                     1,
                     true,
                     true,
                     true,
                     true);
  AssignFile(InputFile,InputFileName);
  Reset(InputFile,1);
  ColCount := (FileSize(InputFile)-16) div (FileSize(LabelFile)-8);
  Close(LabelFile);
  Seek(InputFile,16);  // Skip first 16 bytes in file -- see file format
  SetLength(LoadBytes, RowCount*ColCount); // The actual RowCount, if it is reduced above.
  BlockRead(InputFile, LoadBytes[0], Length(LoadBytes));
  SetLength(LoadSingles, Length(LoadBytes));
  for i:=0 to length(LoadBytes)-1 do
    LoadSingles[i]:=LoadBytes[i]/256.0; // Convert byte to single and rescale to 0..1
  InputT:=CreateTensorSingle([RowCount, ColCount], LoadSingles);
  Close(InputFile);
  SetLength(LoadBytes,0);
  SetLength(LoadSingles,0);
  end;
procedure DestroyInputAndLabel;
  begin
  TF_DeleteTensor(LabelT);
  TF_DeleteTensor(InputT);
  end;

procedure CreateGraphAndSession;
  begin
  // This is the whole logic of the model
  G.init;
  // All the inputs
  G.AddInput('data',TF_FLOAT);        // The picture data
  G.AddInput('label',TF_FLOAT);       // The labels to them
  G.AddInput('batchsize',TF_FLOAT);   // Number of smaples
  G.AddInput('w1o',TF_FLOAT);         // The 6 paramters of the model
  G.AddInput('b1o',TF_FLOAT);
  G.AddInput('w2o',TF_FLOAT);
  G.AddInput('b2o',TF_FLOAT);
  G.AddInput('w3o',TF_FLOAT);
  G.AddInput('b3o',TF_FLOAT);
  G.AddInput('dw1',TF_FLOAT);         // Their deltas
  G.AddInput('db1',TF_FLOAT);
  G.AddInput('dw2',TF_FLOAT);
  G.AddInput('db2',TF_FLOAT);
  G.AddInput('dw3',TF_FLOAT);
  G.AddInput('db3',TF_FLOAT);
  G.AddInput('lastcost',TF_FLOAT);    // To decide whether to approve or not the new model (helps to avoid calculating dw, db unnecessariy)
  G.AddInput('learningrate',TF_FLOAT);// The learning rate
  // Constants
  G.AddConstant('intzero',Int32(0));
  G.AddConstant('singleone',1.0);
  G.AddConstant('singleminusone',-1.0);
  G.AddConstant('floatcorrection',0.0001);
  G.AddTensor('perm',CreateTensorInt32([2],[1,0]),true);
  G.AddTensor('reductions',CreateTensorInt32([2],[0,1]),true);
  // The forward calculation
  G.AddMul('learningrate','dw1','dw1u',TF_FLOAT); // The deltas with the learning rate, used to U_pdate the model
  G.AddMul('learningrate','db1','db1u',TF_FLOAT);
  G.AddMul('learningrate','dw2','dw2u',TF_FLOAT);
  G.AddMul('learningrate','db2','db2u',TF_FLOAT);
  G.AddMul('learningrate','dw3','dw3u',TF_FLOAT);
  G.AddMul('learningrate','db3','db3u',TF_FLOAT);

  G.AddSub('w1o','dw1u','w1t',TF_FLOAT); // The update of the O_ld model to get the model for T_esting
  G.AddSub('b1o','db1u','b1t',TF_FLOAT);
  G.AddSub('w2o','dw2u','w2t',TF_FLOAT);
  G.AddSub('b2o','db2u','b2t',TF_FLOAT);
  G.AddSub('w3o','dw3u','w3t',TF_FLOAT);
  G.AddSub('b3o','db3u','b3t',TF_FLOAT);

  G.AddMatMul('data','w1t','o1',false,false,TF_FLOAT); // The forward propagation
  G.AddAdd('o1','b1t','q1',TF_FLOAT);
  G.AddSigmoid('q1','a1',TF_FLOAT);
  G.AddMatMul('a1','w2t','o2',false,false,TF_FLOAT);
  G.AddAdd('o2','b2t','q2',TF_FLOAT);
  G.AddSigmoid('q2','a2',TF_FLOAT);
  G.AddMatMul('a2','w3t','o3',false,false,TF_FLOAT);
  G.AddAdd('o3','b3t','q3',TF_FLOAT);
  G.AddSoftMax('q3','a3',TF_FLOAT);                    // a3 is the predictions tensor
  // The Cost
  G.AddAdd('a3','floatcorrection','a3corr',TF_FLOAT);  // need to avoid log crash when close to zero
  G.AddLog('a3corr','a3label',TF_FLOAT);               // The crossentropy cost function (could use xlogy as well to save a step)
  G.AddMul('label','a3label','cost-1',TF_FLOAT);
  G.AddSum('cost-1','reductions','cost-2',false,TF_FLOAT,TF_INT32); // Adding up the costs
  G.AddMul('cost-2','singleminusone','cost-3',TF_FLOAT); // Cost must be positive
  G.AddDiv('cost-3','batchsize','actualcost',TF_FLOAT);  // Average cost
  G.AddLess('actualcost','lastcost','approve',TF_FLOAT); // If better than the previous, then delatas are calculated
  // The deltas (using elementary staps, that can be replaced by built in operations at some places. Used if 'approve'
  G.AddSub('a3','label','da3',TF_FLOAT); // The difference between predictions and labels
  G.AddDiv('da3','batchsize','da3n',TF_FLOAT); // Normed
  G.AddIdentity('da3n','do3',TF_FLOAT); // For SoftMax activation, no need to do this step, but to keep the "naming" standard
  G.AddMatMul('do3','a2','dw3t',true,false,TF_FLOAT); // For speed reasons it is done in this "strange" sequence in two steps
  G.AddTranspose('dw3t','perm','dw3c',TF_FLOAT,TF_INT32);
  G.AddSum('do3','intzero','db3c',false,TF_FLOAT,TF_INT32); // delta B C_alculated
  G.AddMatMul('do3','w3t','da2',false,true,TF_FLOAT); // On earlier layers we start with this
  G.AddMul('da2','a2','do2-1',TF_FLOAT); // The sigmoid differnetial y*(1-y)
  G.AddSub('singleone','a2','do2-2',TF_FLOAT);
  G.AddMul('do2-1','do2-2','do2',TF_FLOAT);;
  G.AddMatMul('do2','a1','dw2t',true,false,TF_FLOAT);
  G.AddTranspose('dw2t','perm','dw2c',TF_FLOAT,TF_INT32);
  G.AddSum('do2','intzero','db2c',false,TF_FLOAT,TF_INT32);
  G.AddMatMul('do2','w2t','da1',false,true,TF_FLOAT);
  G.AddMul('da1','a1','do1-1',TF_FLOAT);
  G.AddSub('singleone','a1','do1-2',TF_FLOAT);
  G.AddMul('do1-1','do1-2','do1',TF_FLOAT);;
  G.AddMatMul('do1','data','dw1t',true,false,TF_FLOAT);
  G.AddTranspose('dw1t','perm','dw1c',TF_FLOAT,TF_INT32);
  G.AddSum('do1','intzero','db1c',false,TF_FLOAT,TF_INT32);
  // Update - only if approve. Then the N_ew deltas are the calculated deltas, otherwise the input. The N_ew model is either the
  // T_ested model or the O_ld model
  G.AddSelect('approve','dw1c','dw1','dw1n',TF_FLOAT);
  G.AddSelect('approve','db1c','db1','db1n',TF_FLOAT);
  G.AddSelect('approve','dw2c','dw2','dw2n',TF_FLOAT);
  G.AddSelect('approve','db2c','db2','db2n',TF_FLOAT);
  G.AddSelect('approve','dw3c','dw3','dw3n',TF_FLOAT);
  G.AddSelect('approve','db3c','db3','db3n',TF_FLOAT);
  G.AddSelect('approve','w1t','w1o','w1n',TF_FLOAT);
  G.AddSelect('approve','b1t','b1o','b1n',TF_FLOAT);
  G.AddSelect('approve','w2t','w2o','w2n',TF_FLOAT);
  G.AddSelect('approve','b2t','b2o','b2n',TF_FLOAT);
  G.AddSelect('approve','w3t','w3o','w3n',TF_FLOAT);
  G.AddSelect('approve','b3t','b3o','b3n',TF_FLOAT);
  S.Init(G);
  end;
procedure DestroyGraphAndSession;
  begin
  S.Done;
  G.Done;
  end;
function CalculateAverage(const AInputT:TF_TensorPtr):TF_TensorPtr;
  begin
  result:=ExecSum(AInputT,CreateTEnsorInt32(0),false,false,true);
  result:=ExecDiv(result,CreateTensorSingle(1.0*GetTensorShape(AInputT)[0]),true,true);
  end;
function CreateLayer(const ALayerID:integer;var AInputT:TF_TensorPtr;const AOutputLength:integer;const AAF:TNeuronActivationFunction):TF_TensorPtr;
  var
    TempT:TF_TensorPtr;
    AverageValue:single=0;
    StandardDeviation:single;
  begin
  case AAF of
    nafNone:
      begin
      AverageValue:=0.5;
      StandardDeviation:=0.1;
      end;
    nafSigmoid:
      begin
      AverageValue:=0.0;
      StandardDeviation:=cSigmoidSensitiveRange;
      end;
    nafReLU:
      begin
      AverageValue:=cReLUAverageValue;
      StandardDeviation:=cReLUStandardDeviation;
      end;
    nafTanH:
      begin
      AverageValue:=0.0;
      StandardDeviation:=cTanHSensitiveRange;
      end;
    nafSoftmax:
      begin
      AverageValue:=0.5;
      StandardDeviation:=0.1;
      end;
    // TODO add more if used
    end; // case
  with NeuronLayers[ALayerID] do
    begin
    // Create a random W
    W:=ExecRandomUniform(CreateTensorInt32([2],[GetTensorShape(AInputT)[1],AOutputLength]),Random(1000),Random(1000),TF_FLOAT,true);
    // Norm W and B to get the numbers in the most sensitive area, not to "loose" neurons too early
    result:=ExecMatMul(AInputT,W,false,false);    // would be result if B=0;
    TempT:=CalculateAverage(result);
    result:=ExecSub(result,TempT,true,false);
    result:=ExecSquare(result,true);
    B:=ExecSum(result,CreateTensorInt32(0),false,true);
    B:=ExecDiv(B,CreateTensorSingle(GetTensorShape(AInputT)[0]-1.0),true,true);
    B:=ExecSqrt(B,true); // The standard deviation
    B:=ExecDiv(B,CreateTensorSingle(StandardDeviation),true,true);
    W:=ExecDiv(W,B,true,false);
    B:=ExecDiv(TempT,B,true,true);
    B:=ExecMul(B,CreateTensorSingle(-1.0),true,true);
    B:=ExecAdd(B,CreateTensorSingle(AverageValue),true,true);
    B:=ExecAdd(B,CreateTensorSingleRandom([AOutputLength],-StandardDeviation,StandardDeviation),true,true); // This makes it random
    AF:=AAF;
    result:=ExecMatMul(AInputT,W,false,false);
    result:=ExecAdd(result,B,true);
    end;
  end;
procedure CreateModel(AOriginalLearningRate:Single=cInitialLearningRate);
  var
    IT:TF_TensorPtr;
    OT:TF_TensorPtr;
  begin
  SetLength(NeuronLayers,3);
  OT:=CreateLayer(0,InputT,100,nafSigmoid);

  IT:=OT;
  OT:=CreateLayer(1,IT,30,nafSigmoid);
  TF_DeleteTEnsor(IT);

  IT:=OT;
  OT:=CreateLayer(2,IT,10,nafSoftmax);
  TF_DeleteTEnsor(IT);

  TF_DeleteTensor(OT);
  LearningRate:=AOriginalLearningRate;
  end;
procedure LoadModel;
  var
    ModelFile:file;
    NumberOfLayers:Int64=0;
    InputLength:Int64=0;
    OutputLength:Int64=0;
    ModelSingles:array of Single;
    i:integer;
  begin
  AssignFile(ModelFile,ModelFileName);
  Reset(ModelFile,1);
  BlockRead(ModelFile,NumberOfLayers,SizeOf(NumberOfLayers));
  SetLength(NeuronLayers,NumberOfLayers);
  for i:=0 to NumberOfLayers-1 do
    with NeuronLayers[i] do
      begin
      BlockRead(ModelFile,InputLength,SizeOf(inputLength));
      BlockRead(ModelFile,OutputLength,SizeOf(OutputLength));
      SetLength(ModelSingles,InputLength*OutputLength);
      BlockRead(ModelFile,ModelSingles[0],Length(ModelSingles)*SizeOf(Single));
      W:=CreateTensorSingle([InputLength,OutputLength],ModelSingles);
      SetLength(ModelSingles,OutputLength);
      BlockRead(ModelFile,ModelSingles[0],Length(ModelSingles)*SizeOf(Single));
      B:=CreateTensorSingle([OutputLength],ModelSingles);
      BlockRead(ModelFile,AF,SizeOf(AF));
      end;
  BlockRead(ModelFile,LearningRate,SizeOf(LearningRate));
  Close(ModelFile);
  end;
procedure SaveModel;
  var
    ModelFile:file;
    NumberOfLayers:Int64;
    InputLength:Int64;
    OutputLength:Int64;
    ModelBytes:array of byte;
    i:integer;
  begin
    AssignFile(ModelFile,ModelFileName);
    Rewrite(ModelFile,1);
    NumberOfLayers:=length(NeuronLayers);
    BlockWrite(ModelFile,NumberOfLayers,SizeOf(NumberOfLayers));
    for i:=0 to NumberOfLayers-1 do
      with NeuronLayers[i] do
        begin
          InputLength:=GetTensorShape(W)[0];
          BlockWrite(ModelFile,InputLength,SizeOf(inputLength));
          OutputLength:=GetTensorShape(W)[1];
          BlockWrite(ModelFile,OutputLength,SizeOf(OutputLength));
          OutputLength:=OutputLength * SizeOf(Single); // storing the B length in bytes now
          InputLength:=InputLength*OutputLength;       // storing the W length in bytes now
          SetLength(ModelBytes,InputLength);
          ReadTensorData(W,ModelBytes[0],InputLength);
          BlockWrite(ModelFile,ModelBytes[0],InputLength);
          SetLength(ModelBytes,OutputLength);
          ReadTensorData(B,ModelBytes[0],OutputLength);
          BlockWrite(ModelFile,ModelBytes[0],OutputLength);
          BlockWrite(ModelFile,AF,SizeOf(AF));
        end;
    BlockWrite(ModelFile,LearningRate,SizeOf(LearningRate));
    Close(ModelFile);
  end;
procedure DeleteModel;
  var
    i:integer;
  begin
    for i:=0 to length(NeuronLayers)-1 do
      begin
        with NeuronLayers[i] do
          begin
            TF_DeleteTensor(W);
            TF_DeleteTensor(B);
          end;
      end;
  end;
procedure DestroyModel;
  begin
  DeleteModel;
  SetLength(NeuronLayers,0);
  end;

procedure LearnModel(AIterationCount : Integer);
  var
    ActualCost:Single=0;
    LastCost:real;
    LastReportedCost:Single;
    i:integer;
    IterationCount:Integer=0;
    ImprovementCount:Integer=0;
    ImproveRate:single=0;
    ImprovementDirection:single=1;
    TIns,TOuts:TF_TensorPtrs;
    DebugOptions:TF_StringList;
  begin
  // Prepare the Input Tensors
  SetLength(TIns,17);
  TIns[0]:=InputT;
  Tins[1]:=LabelT;
  TIns[2]:=CreateTensorSingle(GetTensorShape(InputT)[0]);
  TIns[3]:=NeuronLayers[0].w;
  TIns[4]:=NeuronLayers[0].b;
  TIns[5]:=NeuronLayers[1].w;
  TIns[6]:=NeuronLayers[1].b;
  TIns[7]:=NeuronLayers[2].w;
  TIns[8]:=NeuronLayers[2].b;
  SetLength(DebugOptions,0);
  for i:=9 to 14 do
    TIns[i]:=ExecCopy(TIns[i-6],'',DebugOptions,false);
  TIns[15]:=CreateTensorSingle(1000.0);
  TIns[16]:=CreateTensorSingle(0.0); // no d yet, so no ll needed
  TOuts:=S.Run(['data','label','batchsize',
                'w1o','b1o','w2o','b2o','w3o','b3o',
                'dw1','db1','dw2','db2','dw3','db3',
                'lastcost','learningrate'],
               TIns,
               ['w1n','b1n','w2n','b2n','w3n','b3n',
                'dw1n','db1n','dw2n','db2n','dw3n','db3n',
                'actualcost']);
  ReadTensorData(TOuts[12],ActualCost,SizeOf(Single));
  for i:=0 to 12 do
    TIns[3+i]:=TOuts[i];
  LastCost:=ActualCost;
  LastReportedCost:=ActualCost;
  writeln('Starting (random generated or loaded) model gives cost : ',actualcost:8:4,
          ' (total random would give -ln(0.1) = 2.3026, so check it compared to this)');
  SetLength(TOuts,0);
  repeat
    inc(IterationCount);
    TIns[16]:=CreateTensorSingle(LearningRate);
    TOuts:=S.Run(['data','label','batchsize',
                  'w1o','b1o','w2o','b2o','w3o','b3o',
                  'dw1','db1','dw2','db2','dw3','db3',
                  'lastcost','learningrate'],
                 TIns,
                 ['w1n','b1n','w2n','b2n','w3n','b3n',
                  'dw1n','db1n','dw2n','db2n','dw3n','db3n',
                  'actualcost']);

    ReadTensorData(TOuts[12],ActualCost,SizeOf(Single));
    if ActualCost<LastCost then
      begin
      inc(ImprovementCount);
      DeleteModel; // for save, actually it is not needed elsewhere
      NeuronLayers[0].w:=TIns[3];
      NeuronLayers[0].b:=TIns[4];
      NeuronLayers[1].w:=TIns[5];
      NeuronLayers[1].b:=TIns[6];
      NeuronLayers[2].w:=TIns[7];
      NeuronLayers[2].b:=TIns[8];
      for i:=9 to 16 do TF_DeleteTensor(TIns[i]);
      for i:=0 to 12 do TIns[i+3]:=TOuts[i];
      // Fine tune the learning rate to have the maximum impact
      if ImproveRate>0 then // can adjust
        begin
        if ActualCost/LastCost>ImproveRate then
          ImprovementDirection:=-ImprovementDirection;
        if ActualCost/LastCost<>ImproveRate then
          begin
          LearningRate:=LearningRate+ImprovementDirection*LearningRate*cLearningRateAdjustRate;
          if LearningRate>cLearningRateMaximum then LearningRate:=cLearningRateMaximum;
          end;
        end;
      ImproveRate:=ActualCost/LastCost;
      LastCost:=ActualCost;
      end
    else
      begin
      // failed update, need to cut back the learning rate
      for i:=0 to 12 do TF_DeleteTensor(Touts[i]);
      LearningRate:=LearningRate*cLearningRateDecreaseRate;
      ImproveRate:=0;
      end;
    SetLength(TOuts,0);
    if (IterationCount mod cLearningRateReviewFrequency =0) or (IterationCount>=AIterationCount) then
      begin
      Writeln(TimeToStr(Now),' Completed run ',IterationCount:4,' Improved ',ImprovementCount:4,' learningRate ',100*LearningRate:8:4,
              '% Improve rate ',100*(LastReportedCost-ActualCost)/LastReportedCost:8:4,'% Cost ',ActualCost:8:4);
      LastReportedCost:=ActualCost;
      SaveModel;
      end;
    until (ActualCost<=0.1) or (IterationCount>=AIterationCount);
  writeln('FINISHED');
  end;  // LearnModel
procedure RunModel;
  var
    ActualCost         : Single=0;
    TIns, TOuts        : TF_TensorPtrs;
    DebugOptions       : TF_StringList;
    i                  : integer;
    t                  : TF_TensorPtr;
  begin
  // Prepare the Input Tensors
  SetLength(TIns,17);
  TIns[0]:=InputT;
  Tins[1]:=LabelT;
  TIns[2]:=CreateTensorSingle(GetTensorShape(InputT)[0]);
  TIns[3]:=NeuronLayers[0].w;
  TIns[4]:=NeuronLayers[0].b;
  TIns[5]:=NeuronLayers[1].w;
  TIns[6]:=NeuronLayers[1].b;
  TIns[7]:=NeuronLayers[2].w;
  TIns[8]:=NeuronLayers[2].b;
  SetLength(DebugOptions,0);
  for i:=9 to 14 do
    TIns[i]:=ExecCopy(TIns[i-6],'',DebugOptions,false);
  TIns[15]:=CreateTensorSingle(1000.0);
  TIns[16]:=CreateTensorSingle(0.0); // no learning
  TOuts:=S.Run(['data','label','batchsize',
                'w1o','b1o','w2o','b2o','w3o','b3o',
                'dw1','db1','dw2','db2','dw3','db3',
                'lastcost','learningrate'],
               TIns,
               ['actualcost','a3']); // The cost and the predictions only
  ReadTensorData(TOuts[0],ActualCost,SizeOf(Single));
  Writeln('Analysis finished with Cost: ',ActualCost:8:4);
  TF_DeleteTensor(Touts[0]);
  t:=ExecMul(TOuts[1],LabelT,true);
  t:=ExecSum(t,CreateTensorInt32([1],[1]),false,true,true);
  t:=ExecHistogramFixedWidth(t,CreateTensorSingle([2],[0,1]),CreateTensorInt32(10),TF_Int32,true,true,true);
  t:=ExecCast(t,TF_FLOAT,false,true);
  t:=ExecDiv(t,TIns[2],true,true);
  PrintTensorData(t,'The 0-10-20-30-40-50-60-70-80-90 % buckets:');
  TF_DeleteTensor(t);
  SetLength(Tins,0);
  SetLength(TOuts,0);
  end;

begin
randomize;
CreateGraphAndSession;

LoadInputAndLabel(LearnInputFileName,LearnLabelFileName);
CreateModel; // Or comment it and uncomment the next line to load a working model for repeated learning
//  LoadModel;
LearnModel(5000); // The max iteration
RunModel; // A control run on the learning set
DestroyModel;
DestroyInputAndLabel;

LoadInputAndLabel(TestInputFileName,TestLabelFileName);
LoadModel;
RunModel;
DestroyModel;
DestroyInputAndLabel;

DestroyGraphAndSession;
end.

