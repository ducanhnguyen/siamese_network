??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??

x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	? *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	? *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	? *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0

NoOpNoOp
?H
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?G
value?GB?G B?G
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
 
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
?
.iter

/beta_1

0beta_2
	1decay
2learning_ratem?m?"m?#m?(m?)m?3m?4m?5m?6m?7m?8m?v?v?"v?#v?(v?)v?3v?4v?5v?6v?7v?8v?
V
30
41
52
63
74
85
6
7
"8
#9
(10
)11
V
30
41
52
63
74
85
6
7
"8
#9
(10
)11
 
?
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
		variables

trainable_variables
regularization_losses
 
 
h

3kernel
4bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
h

5kernel
6bias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
R
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
R
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
h

7kernel
8bias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
*
30
41
52
63
74
85
*
30
41
52
63
74
85
 
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
 regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
$	variables
%trainable_variables
&regularization_losses
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
*	variables
+trainable_variables
,regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
1
0
1
2
3
4
5
6

k0
l1
 
 

30
41

30
41
 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
>	variables
?trainable_variables
@regularization_losses

50
61

50
61
 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
 
 
 
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
 
 
 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses

70
81

70
81
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_ImageA_InputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
serving_default_ImageB_InputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_ImageA_Inputserving_default_ImageB_Inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_76098
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_76711
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biastotalcounttotal_1count_1Adam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_76856??	
?
r
F__inference_concatenate_layer_call_and_return_conditional_losses_76401
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????? :????????? :Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?	
?
*__inference_JointModel_layer_call_fn_76332

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	? 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
I
-__inference_max_pooling2d_layer_call_fn_76506

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75447?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_75468

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
H__inference_siamese_model_layer_call_and_return_conditional_losses_75783

inputs
inputs_1*
jointmodel_75702:
jointmodel_75704:*
jointmodel_75706:
jointmodel_75708:#
jointmodel_75710:	? 
jointmodel_75712: 
dense_1_75743:@
dense_1_75745:
dense_2_75760:
dense_2_75762:
dense_3_75777:
dense_3_75779:
identity??"JointModel/StatefulPartitionedCall?$JointModel/StatefulPartitionedCall_1?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
"JointModel/StatefulPartitionedCallStatefulPartitionedCallinputsjointmodel_75702jointmodel_75704jointmodel_75706jointmodel_75708jointmodel_75710jointmodel_75712*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75523?
$JointModel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1jointmodel_75702jointmodel_75704jointmodel_75706jointmodel_75708jointmodel_75710jointmodel_75712*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75523?
concatenate/PartitionedCallPartitionedCall+JointModel/StatefulPartitionedCall:output:0-JointModel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_75729?
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_75743dense_1_75745*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75742?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_75760dense_2_75762*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_75759?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_75777dense_3_75779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_75776w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^JointModel/StatefulPartitionedCall%^JointModel/StatefulPartitionedCall_1 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 2H
"JointModel/StatefulPartitionedCall"JointModel/StatefulPartitionedCall2L
$JointModel/StatefulPartitionedCall_1$JointModel/StatefulPartitionedCall_12B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_JointModel_layer_call_and_return_conditional_losses_75672
input_1&
conv2d_75654:
conv2d_75656:(
conv2d_1_75659:
conv2d_1_75661:
dense_75666:	? 
dense_75668: 
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_75654conv2d_75656*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_75468?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_75659conv2d_1_75661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75485?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75495?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75503?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_75666dense_75668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75516u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_76461

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75447

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_conv2d_layer_call_fn_76470

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_75468w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_76532

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_75759

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76521

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_75503

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
p
F__inference_concatenate_layer_call_and_return_conditional_losses_75729

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????? :????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_75776

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_76501

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?a
?
H__inference_siamese_model_layer_call_and_return_conditional_losses_76228
inputs_0
inputs_1J
0jointmodel_conv2d_conv2d_readvariableop_resource:?
1jointmodel_conv2d_biasadd_readvariableop_resource:L
2jointmodel_conv2d_1_conv2d_readvariableop_resource:A
3jointmodel_conv2d_1_biasadd_readvariableop_resource:B
/jointmodel_dense_matmul_readvariableop_resource:	? >
0jointmodel_dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??(JointModel/conv2d/BiasAdd/ReadVariableOp?*JointModel/conv2d/BiasAdd_1/ReadVariableOp?'JointModel/conv2d/Conv2D/ReadVariableOp?)JointModel/conv2d/Conv2D_1/ReadVariableOp?*JointModel/conv2d_1/BiasAdd/ReadVariableOp?,JointModel/conv2d_1/BiasAdd_1/ReadVariableOp?)JointModel/conv2d_1/Conv2D/ReadVariableOp?+JointModel/conv2d_1/Conv2D_1/ReadVariableOp?'JointModel/dense/BiasAdd/ReadVariableOp?)JointModel/dense/BiasAdd_1/ReadVariableOp?&JointModel/dense/MatMul/ReadVariableOp?(JointModel/dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
'JointModel/conv2d/Conv2D/ReadVariableOpReadVariableOp0jointmodel_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
JointModel/conv2d/Conv2DConv2Dinputs_0/JointModel/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
(JointModel/conv2d/BiasAdd/ReadVariableOpReadVariableOp1jointmodel_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
JointModel/conv2d/BiasAddBiasAdd!JointModel/conv2d/Conv2D:output:00JointModel/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
JointModel/conv2d/ReluRelu"JointModel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
)JointModel/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2jointmodel_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
JointModel/conv2d_1/Conv2DConv2D$JointModel/conv2d/Relu:activations:01JointModel/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
*JointModel/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3jointmodel_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
JointModel/conv2d_1/BiasAddBiasAdd#JointModel/conv2d_1/Conv2D:output:02JointModel/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
JointModel/conv2d_1/ReluRelu$JointModel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
 JointModel/max_pooling2d/MaxPoolMaxPool&JointModel/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
i
JointModel/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
JointModel/flatten/ReshapeReshape)JointModel/max_pooling2d/MaxPool:output:0!JointModel/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
&JointModel/dense/MatMul/ReadVariableOpReadVariableOp/jointmodel_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
JointModel/dense/MatMulMatMul#JointModel/flatten/Reshape:output:0.JointModel/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
'JointModel/dense/BiasAdd/ReadVariableOpReadVariableOp0jointmodel_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
JointModel/dense/BiasAddBiasAdd!JointModel/dense/MatMul:product:0/JointModel/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
JointModel/dense/ReluRelu!JointModel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
)JointModel/conv2d/Conv2D_1/ReadVariableOpReadVariableOp0jointmodel_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
JointModel/conv2d/Conv2D_1Conv2Dinputs_11JointModel/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
*JointModel/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp1jointmodel_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
JointModel/conv2d/BiasAdd_1BiasAdd#JointModel/conv2d/Conv2D_1:output:02JointModel/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
JointModel/conv2d/Relu_1Relu$JointModel/conv2d/BiasAdd_1:output:0*
T0*/
_output_shapes
:??????????
+JointModel/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp2jointmodel_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
JointModel/conv2d_1/Conv2D_1Conv2D&JointModel/conv2d/Relu_1:activations:03JointModel/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
,JointModel/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp3jointmodel_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
JointModel/conv2d_1/BiasAdd_1BiasAdd%JointModel/conv2d_1/Conv2D_1:output:04JointModel/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
JointModel/conv2d_1/Relu_1Relu&JointModel/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:??????????
"JointModel/max_pooling2d/MaxPool_1MaxPool(JointModel/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
k
JointModel/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
JointModel/flatten/Reshape_1Reshape+JointModel/max_pooling2d/MaxPool_1:output:0#JointModel/flatten/Const_1:output:0*
T0*(
_output_shapes
:???????????
(JointModel/dense/MatMul_1/ReadVariableOpReadVariableOp/jointmodel_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
JointModel/dense/MatMul_1MatMul%JointModel/flatten/Reshape_1:output:00JointModel/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
)JointModel/dense/BiasAdd_1/ReadVariableOpReadVariableOp0jointmodel_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
JointModel/dense/BiasAdd_1BiasAdd#JointModel/dense/MatMul_1:product:01JointModel/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? v
JointModel/dense/Relu_1Relu#JointModel/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:????????? Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2#JointModel/dense/Relu:activations:0%JointModel/dense/Relu_1:activations:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^JointModel/conv2d/BiasAdd/ReadVariableOp+^JointModel/conv2d/BiasAdd_1/ReadVariableOp(^JointModel/conv2d/Conv2D/ReadVariableOp*^JointModel/conv2d/Conv2D_1/ReadVariableOp+^JointModel/conv2d_1/BiasAdd/ReadVariableOp-^JointModel/conv2d_1/BiasAdd_1/ReadVariableOp*^JointModel/conv2d_1/Conv2D/ReadVariableOp,^JointModel/conv2d_1/Conv2D_1/ReadVariableOp(^JointModel/dense/BiasAdd/ReadVariableOp*^JointModel/dense/BiasAdd_1/ReadVariableOp'^JointModel/dense/MatMul/ReadVariableOp)^JointModel/dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 2T
(JointModel/conv2d/BiasAdd/ReadVariableOp(JointModel/conv2d/BiasAdd/ReadVariableOp2X
*JointModel/conv2d/BiasAdd_1/ReadVariableOp*JointModel/conv2d/BiasAdd_1/ReadVariableOp2R
'JointModel/conv2d/Conv2D/ReadVariableOp'JointModel/conv2d/Conv2D/ReadVariableOp2V
)JointModel/conv2d/Conv2D_1/ReadVariableOp)JointModel/conv2d/Conv2D_1/ReadVariableOp2X
*JointModel/conv2d_1/BiasAdd/ReadVariableOp*JointModel/conv2d_1/BiasAdd/ReadVariableOp2\
,JointModel/conv2d_1/BiasAdd_1/ReadVariableOp,JointModel/conv2d_1/BiasAdd_1/ReadVariableOp2V
)JointModel/conv2d_1/Conv2D/ReadVariableOp)JointModel/conv2d_1/Conv2D/ReadVariableOp2Z
+JointModel/conv2d_1/Conv2D_1/ReadVariableOp+JointModel/conv2d_1/Conv2D_1/ReadVariableOp2R
'JointModel/dense/BiasAdd/ReadVariableOp'JointModel/dense/BiasAdd/ReadVariableOp2V
)JointModel/dense/BiasAdd_1/ReadVariableOp)JointModel/dense/BiasAdd_1/ReadVariableOp2P
&JointModel/dense/MatMul/ReadVariableOp&JointModel/dense/MatMul/ReadVariableOp2T
(JointModel/dense/MatMul_1/ReadVariableOp(JointModel/dense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
(__inference_conv2d_1_layer_call_fn_76490

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75485w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_JointModel_layer_call_and_return_conditional_losses_75619

inputs&
conv2d_75601:
conv2d_75603:(
conv2d_1_75606:
conv2d_1_75608:
dense_75613:	? 
dense_75615: 
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_75601conv2d_75603*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_75468?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_75606conv2d_1_75608*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75485?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75495?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75503?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_75613dense_75615*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75516u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?a
?
H__inference_siamese_model_layer_call_and_return_conditional_losses_76298
inputs_0
inputs_1J
0jointmodel_conv2d_conv2d_readvariableop_resource:?
1jointmodel_conv2d_biasadd_readvariableop_resource:L
2jointmodel_conv2d_1_conv2d_readvariableop_resource:A
3jointmodel_conv2d_1_biasadd_readvariableop_resource:B
/jointmodel_dense_matmul_readvariableop_resource:	? >
0jointmodel_dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:8
&dense_3_matmul_readvariableop_resource:5
'dense_3_biasadd_readvariableop_resource:
identity??(JointModel/conv2d/BiasAdd/ReadVariableOp?*JointModel/conv2d/BiasAdd_1/ReadVariableOp?'JointModel/conv2d/Conv2D/ReadVariableOp?)JointModel/conv2d/Conv2D_1/ReadVariableOp?*JointModel/conv2d_1/BiasAdd/ReadVariableOp?,JointModel/conv2d_1/BiasAdd_1/ReadVariableOp?)JointModel/conv2d_1/Conv2D/ReadVariableOp?+JointModel/conv2d_1/Conv2D_1/ReadVariableOp?'JointModel/dense/BiasAdd/ReadVariableOp?)JointModel/dense/BiasAdd_1/ReadVariableOp?&JointModel/dense/MatMul/ReadVariableOp?(JointModel/dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
'JointModel/conv2d/Conv2D/ReadVariableOpReadVariableOp0jointmodel_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
JointModel/conv2d/Conv2DConv2Dinputs_0/JointModel/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
(JointModel/conv2d/BiasAdd/ReadVariableOpReadVariableOp1jointmodel_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
JointModel/conv2d/BiasAddBiasAdd!JointModel/conv2d/Conv2D:output:00JointModel/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
JointModel/conv2d/ReluRelu"JointModel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
)JointModel/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2jointmodel_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
JointModel/conv2d_1/Conv2DConv2D$JointModel/conv2d/Relu:activations:01JointModel/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
*JointModel/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3jointmodel_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
JointModel/conv2d_1/BiasAddBiasAdd#JointModel/conv2d_1/Conv2D:output:02JointModel/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
JointModel/conv2d_1/ReluRelu$JointModel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
 JointModel/max_pooling2d/MaxPoolMaxPool&JointModel/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
i
JointModel/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
JointModel/flatten/ReshapeReshape)JointModel/max_pooling2d/MaxPool:output:0!JointModel/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
&JointModel/dense/MatMul/ReadVariableOpReadVariableOp/jointmodel_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
JointModel/dense/MatMulMatMul#JointModel/flatten/Reshape:output:0.JointModel/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
'JointModel/dense/BiasAdd/ReadVariableOpReadVariableOp0jointmodel_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
JointModel/dense/BiasAddBiasAdd!JointModel/dense/MatMul:product:0/JointModel/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
JointModel/dense/ReluRelu!JointModel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
)JointModel/conv2d/Conv2D_1/ReadVariableOpReadVariableOp0jointmodel_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
JointModel/conv2d/Conv2D_1Conv2Dinputs_11JointModel/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
*JointModel/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp1jointmodel_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
JointModel/conv2d/BiasAdd_1BiasAdd#JointModel/conv2d/Conv2D_1:output:02JointModel/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
JointModel/conv2d/Relu_1Relu$JointModel/conv2d/BiasAdd_1:output:0*
T0*/
_output_shapes
:??????????
+JointModel/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp2jointmodel_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
JointModel/conv2d_1/Conv2D_1Conv2D&JointModel/conv2d/Relu_1:activations:03JointModel/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
,JointModel/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOp3jointmodel_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
JointModel/conv2d_1/BiasAdd_1BiasAdd%JointModel/conv2d_1/Conv2D_1:output:04JointModel/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
JointModel/conv2d_1/Relu_1Relu&JointModel/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:??????????
"JointModel/max_pooling2d/MaxPool_1MaxPool(JointModel/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
k
JointModel/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
JointModel/flatten/Reshape_1Reshape+JointModel/max_pooling2d/MaxPool_1:output:0#JointModel/flatten/Const_1:output:0*
T0*(
_output_shapes
:???????????
(JointModel/dense/MatMul_1/ReadVariableOpReadVariableOp/jointmodel_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
JointModel/dense/MatMul_1MatMul%JointModel/flatten/Reshape_1:output:00JointModel/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
)JointModel/dense/BiasAdd_1/ReadVariableOpReadVariableOp0jointmodel_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
JointModel/dense/BiasAdd_1BiasAdd#JointModel/dense/MatMul_1:product:01JointModel/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? v
JointModel/dense/Relu_1Relu#JointModel/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:????????? Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate/concatConcatV2#JointModel/dense/Relu:activations:0%JointModel/dense/Relu_1:activations:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
dense_1/MatMulMatMulconcatenate/concat:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp)^JointModel/conv2d/BiasAdd/ReadVariableOp+^JointModel/conv2d/BiasAdd_1/ReadVariableOp(^JointModel/conv2d/Conv2D/ReadVariableOp*^JointModel/conv2d/Conv2D_1/ReadVariableOp+^JointModel/conv2d_1/BiasAdd/ReadVariableOp-^JointModel/conv2d_1/BiasAdd_1/ReadVariableOp*^JointModel/conv2d_1/Conv2D/ReadVariableOp,^JointModel/conv2d_1/Conv2D_1/ReadVariableOp(^JointModel/dense/BiasAdd/ReadVariableOp*^JointModel/dense/BiasAdd_1/ReadVariableOp'^JointModel/dense/MatMul/ReadVariableOp)^JointModel/dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 2T
(JointModel/conv2d/BiasAdd/ReadVariableOp(JointModel/conv2d/BiasAdd/ReadVariableOp2X
*JointModel/conv2d/BiasAdd_1/ReadVariableOp*JointModel/conv2d/BiasAdd_1/ReadVariableOp2R
'JointModel/conv2d/Conv2D/ReadVariableOp'JointModel/conv2d/Conv2D/ReadVariableOp2V
)JointModel/conv2d/Conv2D_1/ReadVariableOp)JointModel/conv2d/Conv2D_1/ReadVariableOp2X
*JointModel/conv2d_1/BiasAdd/ReadVariableOp*JointModel/conv2d_1/BiasAdd/ReadVariableOp2\
,JointModel/conv2d_1/BiasAdd_1/ReadVariableOp,JointModel/conv2d_1/BiasAdd_1/ReadVariableOp2V
)JointModel/conv2d_1/Conv2D/ReadVariableOp)JointModel/conv2d_1/Conv2D/ReadVariableOp2Z
+JointModel/conv2d_1/Conv2D_1/ReadVariableOp+JointModel/conv2d_1/Conv2D_1/ReadVariableOp2R
'JointModel/dense/BiasAdd/ReadVariableOp'JointModel/dense/BiasAdd/ReadVariableOp2V
)JointModel/dense/BiasAdd_1/ReadVariableOp)JointModel/dense/BiasAdd_1/ReadVariableOp2P
&JointModel/dense/MatMul/ReadVariableOp&JointModel/dense/MatMul/ReadVariableOp2T
(JointModel/dense/MatMul_1/ReadVariableOp(JointModel/dense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
E__inference_JointModel_layer_call_and_return_conditional_losses_76360

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	? 3
%dense_biasadd_readvariableop_resource: 
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_JointModel_layer_call_and_return_conditional_losses_76388

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:7
$dense_matmul_readvariableop_resource:	? 3
%dense_biasadd_readvariableop_resource: 
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
max_pooling2d/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_siamese_model_layer_call_fn_75810
imagea_input
imageb_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	? 
	unknown_4: 
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimagea_inputimageb_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_siamese_model_layer_call_and_return_conditional_losses_75783o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameImageA_Input:]Y
/
_output_shapes
:?????????
&
_user_specified_nameImageB_Input
?
C
'__inference_flatten_layer_call_fn_76526

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75503a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?X
?
__inference__traced_save_76711
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:::::: : : : : :::::	? : : : : : :@::::::::::	? : :@::::::::::	? : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	? : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::% !

_output_shapes
:	? : !

_output_shapes
: :$" 

_output_shapes

:@: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::$& 

_output_shapes

:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::%,!

_output_shapes
:	? : -

_output_shapes
: :.

_output_shapes
: 
?!
?
H__inference_siamese_model_layer_call_and_return_conditional_losses_76019
imagea_input
imageb_input*
jointmodel_75982:
jointmodel_75984:*
jointmodel_75986:
jointmodel_75988:#
jointmodel_75990:	? 
jointmodel_75992: 
dense_1_76003:@
dense_1_76005:
dense_2_76008:
dense_2_76010:
dense_3_76013:
dense_3_76015:
identity??"JointModel/StatefulPartitionedCall?$JointModel/StatefulPartitionedCall_1?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
"JointModel/StatefulPartitionedCallStatefulPartitionedCallimagea_inputjointmodel_75982jointmodel_75984jointmodel_75986jointmodel_75988jointmodel_75990jointmodel_75992*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75523?
$JointModel/StatefulPartitionedCall_1StatefulPartitionedCallimageb_inputjointmodel_75982jointmodel_75984jointmodel_75986jointmodel_75988jointmodel_75990jointmodel_75992*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75523?
concatenate/PartitionedCallPartitionedCall+JointModel/StatefulPartitionedCall:output:0-JointModel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_75729?
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_76003dense_1_76005*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75742?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_76008dense_2_76010*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_75759?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_76013dense_3_76015*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_75776w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^JointModel/StatefulPartitionedCall%^JointModel/StatefulPartitionedCall_1 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 2H
"JointModel/StatefulPartitionedCall"JointModel/StatefulPartitionedCall2L
$JointModel/StatefulPartitionedCall_1$JointModel/StatefulPartitionedCall_12B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameImageA_Input:]Y
/
_output_shapes
:?????????
&
_user_specified_nameImageB_Input
?
I
-__inference_max_pooling2d_layer_call_fn_76511

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75495h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_siamese_model_layer_call_fn_76128
inputs_0
inputs_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	? 
	unknown_4: 
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_siamese_model_layer_call_and_return_conditional_losses_75783o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
%__inference_dense_layer_call_fn_76541

inputs
unknown:	? 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75516o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76516

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
W
+__inference_concatenate_layer_call_fn_76394
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_75729`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:????????? :????????? :Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/1
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75485

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_76410

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75742o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_dense_3_layer_call_fn_76450

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_75776o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?x
?
 __inference__wrapped_model_75438
imagea_input
imageb_inputX
>siamese_model_jointmodel_conv2d_conv2d_readvariableop_resource:M
?siamese_model_jointmodel_conv2d_biasadd_readvariableop_resource:Z
@siamese_model_jointmodel_conv2d_1_conv2d_readvariableop_resource:O
Asiamese_model_jointmodel_conv2d_1_biasadd_readvariableop_resource:P
=siamese_model_jointmodel_dense_matmul_readvariableop_resource:	? L
>siamese_model_jointmodel_dense_biasadd_readvariableop_resource: F
4siamese_model_dense_1_matmul_readvariableop_resource:@C
5siamese_model_dense_1_biasadd_readvariableop_resource:F
4siamese_model_dense_2_matmul_readvariableop_resource:C
5siamese_model_dense_2_biasadd_readvariableop_resource:F
4siamese_model_dense_3_matmul_readvariableop_resource:C
5siamese_model_dense_3_biasadd_readvariableop_resource:
identity??6siamese_model/JointModel/conv2d/BiasAdd/ReadVariableOp?8siamese_model/JointModel/conv2d/BiasAdd_1/ReadVariableOp?5siamese_model/JointModel/conv2d/Conv2D/ReadVariableOp?7siamese_model/JointModel/conv2d/Conv2D_1/ReadVariableOp?8siamese_model/JointModel/conv2d_1/BiasAdd/ReadVariableOp?:siamese_model/JointModel/conv2d_1/BiasAdd_1/ReadVariableOp?7siamese_model/JointModel/conv2d_1/Conv2D/ReadVariableOp?9siamese_model/JointModel/conv2d_1/Conv2D_1/ReadVariableOp?5siamese_model/JointModel/dense/BiasAdd/ReadVariableOp?7siamese_model/JointModel/dense/BiasAdd_1/ReadVariableOp?4siamese_model/JointModel/dense/MatMul/ReadVariableOp?6siamese_model/JointModel/dense/MatMul_1/ReadVariableOp?,siamese_model/dense_1/BiasAdd/ReadVariableOp?+siamese_model/dense_1/MatMul/ReadVariableOp?,siamese_model/dense_2/BiasAdd/ReadVariableOp?+siamese_model/dense_2/MatMul/ReadVariableOp?,siamese_model/dense_3/BiasAdd/ReadVariableOp?+siamese_model/dense_3/MatMul/ReadVariableOp?
5siamese_model/JointModel/conv2d/Conv2D/ReadVariableOpReadVariableOp>siamese_model_jointmodel_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
&siamese_model/JointModel/conv2d/Conv2DConv2Dimagea_input=siamese_model/JointModel/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
6siamese_model/JointModel/conv2d/BiasAdd/ReadVariableOpReadVariableOp?siamese_model_jointmodel_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
'siamese_model/JointModel/conv2d/BiasAddBiasAdd/siamese_model/JointModel/conv2d/Conv2D:output:0>siamese_model/JointModel/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
$siamese_model/JointModel/conv2d/ReluRelu0siamese_model/JointModel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
7siamese_model/JointModel/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@siamese_model_jointmodel_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
(siamese_model/JointModel/conv2d_1/Conv2DConv2D2siamese_model/JointModel/conv2d/Relu:activations:0?siamese_model/JointModel/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
8siamese_model/JointModel/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpAsiamese_model_jointmodel_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)siamese_model/JointModel/conv2d_1/BiasAddBiasAdd1siamese_model/JointModel/conv2d_1/Conv2D:output:0@siamese_model/JointModel/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
&siamese_model/JointModel/conv2d_1/ReluRelu2siamese_model/JointModel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
.siamese_model/JointModel/max_pooling2d/MaxPoolMaxPool4siamese_model/JointModel/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
w
&siamese_model/JointModel/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
(siamese_model/JointModel/flatten/ReshapeReshape7siamese_model/JointModel/max_pooling2d/MaxPool:output:0/siamese_model/JointModel/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
4siamese_model/JointModel/dense/MatMul/ReadVariableOpReadVariableOp=siamese_model_jointmodel_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
%siamese_model/JointModel/dense/MatMulMatMul1siamese_model/JointModel/flatten/Reshape:output:0<siamese_model/JointModel/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
5siamese_model/JointModel/dense/BiasAdd/ReadVariableOpReadVariableOp>siamese_model_jointmodel_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
&siamese_model/JointModel/dense/BiasAddBiasAdd/siamese_model/JointModel/dense/MatMul:product:0=siamese_model/JointModel/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
#siamese_model/JointModel/dense/ReluRelu/siamese_model/JointModel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
7siamese_model/JointModel/conv2d/Conv2D_1/ReadVariableOpReadVariableOp>siamese_model_jointmodel_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
(siamese_model/JointModel/conv2d/Conv2D_1Conv2Dimageb_input?siamese_model/JointModel/conv2d/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
8siamese_model/JointModel/conv2d/BiasAdd_1/ReadVariableOpReadVariableOp?siamese_model_jointmodel_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
)siamese_model/JointModel/conv2d/BiasAdd_1BiasAdd1siamese_model/JointModel/conv2d/Conv2D_1:output:0@siamese_model/JointModel/conv2d/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
&siamese_model/JointModel/conv2d/Relu_1Relu2siamese_model/JointModel/conv2d/BiasAdd_1:output:0*
T0*/
_output_shapes
:??????????
9siamese_model/JointModel/conv2d_1/Conv2D_1/ReadVariableOpReadVariableOp@siamese_model_jointmodel_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
*siamese_model/JointModel/conv2d_1/Conv2D_1Conv2D4siamese_model/JointModel/conv2d/Relu_1:activations:0Asiamese_model/JointModel/conv2d_1/Conv2D_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
:siamese_model/JointModel/conv2d_1/BiasAdd_1/ReadVariableOpReadVariableOpAsiamese_model_jointmodel_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
+siamese_model/JointModel/conv2d_1/BiasAdd_1BiasAdd3siamese_model/JointModel/conv2d_1/Conv2D_1:output:0Bsiamese_model/JointModel/conv2d_1/BiasAdd_1/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
(siamese_model/JointModel/conv2d_1/Relu_1Relu4siamese_model/JointModel/conv2d_1/BiasAdd_1:output:0*
T0*/
_output_shapes
:??????????
0siamese_model/JointModel/max_pooling2d/MaxPool_1MaxPool6siamese_model/JointModel/conv2d_1/Relu_1:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
y
(siamese_model/JointModel/flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"???? 	  ?
*siamese_model/JointModel/flatten/Reshape_1Reshape9siamese_model/JointModel/max_pooling2d/MaxPool_1:output:01siamese_model/JointModel/flatten/Const_1:output:0*
T0*(
_output_shapes
:???????????
6siamese_model/JointModel/dense/MatMul_1/ReadVariableOpReadVariableOp=siamese_model_jointmodel_dense_matmul_readvariableop_resource*
_output_shapes
:	? *
dtype0?
'siamese_model/JointModel/dense/MatMul_1MatMul3siamese_model/JointModel/flatten/Reshape_1:output:0>siamese_model/JointModel/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
7siamese_model/JointModel/dense/BiasAdd_1/ReadVariableOpReadVariableOp>siamese_model_jointmodel_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
(siamese_model/JointModel/dense/BiasAdd_1BiasAdd1siamese_model/JointModel/dense/MatMul_1:product:0?siamese_model/JointModel/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
%siamese_model/JointModel/dense/Relu_1Relu1siamese_model/JointModel/dense/BiasAdd_1:output:0*
T0*'
_output_shapes
:????????? g
%siamese_model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
 siamese_model/concatenate/concatConcatV21siamese_model/JointModel/dense/Relu:activations:03siamese_model/JointModel/dense/Relu_1:activations:0.siamese_model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????@?
+siamese_model/dense_1/MatMul/ReadVariableOpReadVariableOp4siamese_model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
siamese_model/dense_1/MatMulMatMul)siamese_model/concatenate/concat:output:03siamese_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,siamese_model/dense_1/BiasAdd/ReadVariableOpReadVariableOp5siamese_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
siamese_model/dense_1/BiasAddBiasAdd&siamese_model/dense_1/MatMul:product:04siamese_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
siamese_model/dense_1/ReluRelu&siamese_model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
+siamese_model/dense_2/MatMul/ReadVariableOpReadVariableOp4siamese_model_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
siamese_model/dense_2/MatMulMatMul(siamese_model/dense_1/Relu:activations:03siamese_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,siamese_model/dense_2/BiasAdd/ReadVariableOpReadVariableOp5siamese_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
siamese_model/dense_2/BiasAddBiasAdd&siamese_model/dense_2/MatMul:product:04siamese_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
siamese_model/dense_2/ReluRelu&siamese_model/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
+siamese_model/dense_3/MatMul/ReadVariableOpReadVariableOp4siamese_model_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
siamese_model/dense_3/MatMulMatMul(siamese_model/dense_2/Relu:activations:03siamese_model/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
,siamese_model/dense_3/BiasAdd/ReadVariableOpReadVariableOp5siamese_model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
siamese_model/dense_3/BiasAddBiasAdd&siamese_model/dense_3/MatMul:product:04siamese_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
siamese_model/dense_3/SigmoidSigmoid&siamese_model/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????p
IdentityIdentity!siamese_model/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp7^siamese_model/JointModel/conv2d/BiasAdd/ReadVariableOp9^siamese_model/JointModel/conv2d/BiasAdd_1/ReadVariableOp6^siamese_model/JointModel/conv2d/Conv2D/ReadVariableOp8^siamese_model/JointModel/conv2d/Conv2D_1/ReadVariableOp9^siamese_model/JointModel/conv2d_1/BiasAdd/ReadVariableOp;^siamese_model/JointModel/conv2d_1/BiasAdd_1/ReadVariableOp8^siamese_model/JointModel/conv2d_1/Conv2D/ReadVariableOp:^siamese_model/JointModel/conv2d_1/Conv2D_1/ReadVariableOp6^siamese_model/JointModel/dense/BiasAdd/ReadVariableOp8^siamese_model/JointModel/dense/BiasAdd_1/ReadVariableOp5^siamese_model/JointModel/dense/MatMul/ReadVariableOp7^siamese_model/JointModel/dense/MatMul_1/ReadVariableOp-^siamese_model/dense_1/BiasAdd/ReadVariableOp,^siamese_model/dense_1/MatMul/ReadVariableOp-^siamese_model/dense_2/BiasAdd/ReadVariableOp,^siamese_model/dense_2/MatMul/ReadVariableOp-^siamese_model/dense_3/BiasAdd/ReadVariableOp,^siamese_model/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 2p
6siamese_model/JointModel/conv2d/BiasAdd/ReadVariableOp6siamese_model/JointModel/conv2d/BiasAdd/ReadVariableOp2t
8siamese_model/JointModel/conv2d/BiasAdd_1/ReadVariableOp8siamese_model/JointModel/conv2d/BiasAdd_1/ReadVariableOp2n
5siamese_model/JointModel/conv2d/Conv2D/ReadVariableOp5siamese_model/JointModel/conv2d/Conv2D/ReadVariableOp2r
7siamese_model/JointModel/conv2d/Conv2D_1/ReadVariableOp7siamese_model/JointModel/conv2d/Conv2D_1/ReadVariableOp2t
8siamese_model/JointModel/conv2d_1/BiasAdd/ReadVariableOp8siamese_model/JointModel/conv2d_1/BiasAdd/ReadVariableOp2x
:siamese_model/JointModel/conv2d_1/BiasAdd_1/ReadVariableOp:siamese_model/JointModel/conv2d_1/BiasAdd_1/ReadVariableOp2r
7siamese_model/JointModel/conv2d_1/Conv2D/ReadVariableOp7siamese_model/JointModel/conv2d_1/Conv2D/ReadVariableOp2v
9siamese_model/JointModel/conv2d_1/Conv2D_1/ReadVariableOp9siamese_model/JointModel/conv2d_1/Conv2D_1/ReadVariableOp2n
5siamese_model/JointModel/dense/BiasAdd/ReadVariableOp5siamese_model/JointModel/dense/BiasAdd/ReadVariableOp2r
7siamese_model/JointModel/dense/BiasAdd_1/ReadVariableOp7siamese_model/JointModel/dense/BiasAdd_1/ReadVariableOp2l
4siamese_model/JointModel/dense/MatMul/ReadVariableOp4siamese_model/JointModel/dense/MatMul/ReadVariableOp2p
6siamese_model/JointModel/dense/MatMul_1/ReadVariableOp6siamese_model/JointModel/dense/MatMul_1/ReadVariableOp2\
,siamese_model/dense_1/BiasAdd/ReadVariableOp,siamese_model/dense_1/BiasAdd/ReadVariableOp2Z
+siamese_model/dense_1/MatMul/ReadVariableOp+siamese_model/dense_1/MatMul/ReadVariableOp2\
,siamese_model/dense_2/BiasAdd/ReadVariableOp,siamese_model/dense_2/BiasAdd/ReadVariableOp2Z
+siamese_model/dense_2/MatMul/ReadVariableOp+siamese_model/dense_2/MatMul/ReadVariableOp2\
,siamese_model/dense_3/BiasAdd/ReadVariableOp,siamese_model/dense_3/BiasAdd/ReadVariableOp2Z
+siamese_model/dense_3/MatMul/ReadVariableOp+siamese_model/dense_3/MatMul/ReadVariableOp:] Y
/
_output_shapes
:?????????
&
_user_specified_nameImageA_Input:]Y
/
_output_shapes
:?????????
&
_user_specified_nameImageB_Input
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75495

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_75742

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
*__inference_JointModel_layer_call_fn_76315

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	? 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75523o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
H__inference_siamese_model_layer_call_and_return_conditional_losses_75921

inputs
inputs_1*
jointmodel_75884:
jointmodel_75886:*
jointmodel_75888:
jointmodel_75890:#
jointmodel_75892:	? 
jointmodel_75894: 
dense_1_75905:@
dense_1_75907:
dense_2_75910:
dense_2_75912:
dense_3_75915:
dense_3_75917:
identity??"JointModel/StatefulPartitionedCall?$JointModel/StatefulPartitionedCall_1?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
"JointModel/StatefulPartitionedCallStatefulPartitionedCallinputsjointmodel_75884jointmodel_75886jointmodel_75888jointmodel_75890jointmodel_75892jointmodel_75894*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75619?
$JointModel/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1jointmodel_75884jointmodel_75886jointmodel_75888jointmodel_75890jointmodel_75892jointmodel_75894*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75619?
concatenate/PartitionedCallPartitionedCall+JointModel/StatefulPartitionedCall:output:0-JointModel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_75729?
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_75905dense_1_75907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75742?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_75910dense_2_75912*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_75759?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_75915dense_3_75917*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_75776w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^JointModel/StatefulPartitionedCall%^JointModel/StatefulPartitionedCall_1 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 2H
"JointModel/StatefulPartitionedCall"JointModel/StatefulPartitionedCall2L
$JointModel/StatefulPartitionedCall_1$JointModel/StatefulPartitionedCall_12B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
@__inference_dense_layer_call_and_return_conditional_losses_75516

inputs1
matmul_readvariableop_resource:	? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
*__inference_JointModel_layer_call_fn_75538
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	? 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75523o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_76421

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
*__inference_JointModel_layer_call_fn_75651
input_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	? 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_76441

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
H__inference_siamese_model_layer_call_and_return_conditional_losses_76060
imagea_input
imageb_input*
jointmodel_76023:
jointmodel_76025:*
jointmodel_76027:
jointmodel_76029:#
jointmodel_76031:	? 
jointmodel_76033: 
dense_1_76044:@
dense_1_76046:
dense_2_76049:
dense_2_76051:
dense_3_76054:
dense_3_76056:
identity??"JointModel/StatefulPartitionedCall?$JointModel/StatefulPartitionedCall_1?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
"JointModel/StatefulPartitionedCallStatefulPartitionedCallimagea_inputjointmodel_76023jointmodel_76025jointmodel_76027jointmodel_76029jointmodel_76031jointmodel_76033*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75619?
$JointModel/StatefulPartitionedCall_1StatefulPartitionedCallimageb_inputjointmodel_76023jointmodel_76025jointmodel_76027jointmodel_76029jointmodel_76031jointmodel_76033*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_JointModel_layer_call_and_return_conditional_losses_75619?
concatenate/PartitionedCallPartitionedCall+JointModel/StatefulPartitionedCall:output:0-JointModel/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_75729?
dense_1/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_1_76044dense_1_76046*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_75742?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_76049dense_2_76051*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_75759?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_76054dense_3_76056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_75776w
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp#^JointModel/StatefulPartitionedCall%^JointModel/StatefulPartitionedCall_1 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 2H
"JointModel/StatefulPartitionedCall"JointModel/StatefulPartitionedCall2L
$JointModel/StatefulPartitionedCall_1$JointModel/StatefulPartitionedCall_12B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameImageA_Input:]Y
/
_output_shapes
:?????????
&
_user_specified_nameImageB_Input
?
?
-__inference_siamese_model_layer_call_fn_76158
inputs_0
inputs_1!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	? 
	unknown_4: 
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_siamese_model_layer_call_and_return_conditional_losses_75921o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
@__inference_dense_layer_call_and_return_conditional_losses_76552

inputs1
matmul_readvariableop_resource:	? -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_JointModel_layer_call_and_return_conditional_losses_75523

inputs&
conv2d_75469:
conv2d_75471:(
conv2d_1_75486:
conv2d_1_75488:
dense_75517:	? 
dense_75519: 
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_75469conv2d_75471*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_75468?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_75486conv2d_1_75488*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75485?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75495?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75503?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_75517dense_75519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75516u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_siamese_model_layer_call_fn_75978
imagea_input
imageb_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	? 
	unknown_4: 
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimagea_inputimageb_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_siamese_model_layer_call_and_return_conditional_losses_75921o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameImageA_Input:]Y
/
_output_shapes
:?????????
&
_user_specified_nameImageB_Input
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_76481

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
ɰ
?
!__inference__traced_restore_76856
file_prefix1
assignvariableop_dense_1_kernel:@-
assignvariableop_1_dense_1_bias:3
!assignvariableop_2_dense_2_kernel:-
assignvariableop_3_dense_2_bias:3
!assignvariableop_4_dense_3_kernel:-
assignvariableop_5_dense_3_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: ;
!assignvariableop_11_conv2d_kernel:-
assignvariableop_12_conv2d_bias:=
#assignvariableop_13_conv2d_1_kernel:/
!assignvariableop_14_conv2d_1_bias:3
 assignvariableop_15_dense_kernel:	? ,
assignvariableop_16_dense_bias: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: ;
)assignvariableop_21_adam_dense_1_kernel_m:@5
'assignvariableop_22_adam_dense_1_bias_m:;
)assignvariableop_23_adam_dense_2_kernel_m:5
'assignvariableop_24_adam_dense_2_bias_m:;
)assignvariableop_25_adam_dense_3_kernel_m:5
'assignvariableop_26_adam_dense_3_bias_m:B
(assignvariableop_27_adam_conv2d_kernel_m:4
&assignvariableop_28_adam_conv2d_bias_m:D
*assignvariableop_29_adam_conv2d_1_kernel_m:6
(assignvariableop_30_adam_conv2d_1_bias_m::
'assignvariableop_31_adam_dense_kernel_m:	? 3
%assignvariableop_32_adam_dense_bias_m: ;
)assignvariableop_33_adam_dense_1_kernel_v:@5
'assignvariableop_34_adam_dense_1_bias_v:;
)assignvariableop_35_adam_dense_2_kernel_v:5
'assignvariableop_36_adam_dense_2_bias_v:;
)assignvariableop_37_adam_dense_3_kernel_v:5
'assignvariableop_38_adam_dense_3_bias_v:B
(assignvariableop_39_adam_conv2d_kernel_v:4
&assignvariableop_40_adam_conv2d_bias_v:D
*assignvariableop_41_adam_conv2d_1_kernel_v:6
(assignvariableop_42_adam_conv2d_1_bias_v::
'assignvariableop_43_adam_dense_kernel_v:	? 3
%assignvariableop_44_adam_dense_bias_v: 
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_conv2d_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_dense_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv2d_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv2d_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_conv2d_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_conv2d_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_dense_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_dense_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_46IdentityIdentity_45:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
'__inference_dense_2_layer_call_fn_76430

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_75759o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_76098
imagea_input
imageb_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
	unknown_3:	? 
	unknown_4: 
	unknown_5:@
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallimagea_inputimageb_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_75438o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:?????????:?????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameImageA_Input:]Y
/
_output_shapes
:?????????
&
_user_specified_nameImageB_Input
?
?
E__inference_JointModel_layer_call_and_return_conditional_losses_75693
input_1&
conv2d_75675:
conv2d_75677:(
conv2d_1_75680:
conv2d_1_75682:
dense_75687:	? 
dense_75689: 
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_75675conv2d_75677*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_75468?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_75680conv2d_1_75682*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_75485?
max_pooling2d/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_75495?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_75503?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_75687dense_75689*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_75516u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? ?
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
ImageA_Input=
serving_default_ImageA_Input:0?????????
M
ImageB_Input=
serving_default_ImageB_Input:0?????????;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_network
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
.iter

/beta_1

0beta_2
	1decay
2learning_ratem?m?"m?#m?(m?)m?3m?4m?5m?6m?7m?8m?v?v?"v?#v?(v?)v?3v?4v?5v?6v?7v?8v?"
	optimizer
v
30
41
52
63
74
85
6
7
"8
#9
(10
)11"
trackable_list_wrapper
v
30
41
52
63
74
85
6
7
"8
#9
(10
)11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
		variables

trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
"
_tf_keras_input_layer
?

3kernel
4bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

5kernel
6bias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

7kernel
8bias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
J
30
41
52
63
74
85"
trackable_list_wrapper
J
30
41
52
63
74
85"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
 regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
$	variables
%trainable_variables
&regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
*	variables
+trainable_variables
,regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%2conv2d/kernel
:2conv2d/bias
):'2conv2d_1/kernel
:2conv2d_1/bias
:	? 2dense/kernel
: 2
dense/bias
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
>	variables
?trainable_variables
@regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
%:#@2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
%:#2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:#2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
$:"	? 2Adam/dense/kernel/m
: 2Adam/dense/bias/m
%:#@2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
%:#2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
%:#2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
$:"	? 2Adam/dense/kernel/v
: 2Adam/dense/bias/v
?2?
-__inference_siamese_model_layer_call_fn_75810
-__inference_siamese_model_layer_call_fn_76128
-__inference_siamese_model_layer_call_fn_76158
-__inference_siamese_model_layer_call_fn_75978?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_siamese_model_layer_call_and_return_conditional_losses_76228
H__inference_siamese_model_layer_call_and_return_conditional_losses_76298
H__inference_siamese_model_layer_call_and_return_conditional_losses_76019
H__inference_siamese_model_layer_call_and_return_conditional_losses_76060?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_75438ImageA_InputImageB_Input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_JointModel_layer_call_fn_75538
*__inference_JointModel_layer_call_fn_76315
*__inference_JointModel_layer_call_fn_76332
*__inference_JointModel_layer_call_fn_75651?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_JointModel_layer_call_and_return_conditional_losses_76360
E__inference_JointModel_layer_call_and_return_conditional_losses_76388
E__inference_JointModel_layer_call_and_return_conditional_losses_75672
E__inference_JointModel_layer_call_and_return_conditional_losses_75693?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_concatenate_layer_call_fn_76394?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_76401?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_76410?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_76421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_76430?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_76441?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_3_layer_call_fn_76450?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_76461?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_76098ImageA_InputImageB_Input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_conv2d_layer_call_fn_76470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_76481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_1_layer_call_fn_76490?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_76501?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_max_pooling2d_layer_call_fn_76506
-__inference_max_pooling2d_layer_call_fn_76511?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76516
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76521?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_76526?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_76532?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_76541?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_76552?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
E__inference_JointModel_layer_call_and_return_conditional_losses_75672q345678@?=
6?3
)?&
input_1?????????
p 

 
? "%?"
?
0????????? 
? ?
E__inference_JointModel_layer_call_and_return_conditional_losses_75693q345678@?=
6?3
)?&
input_1?????????
p

 
? "%?"
?
0????????? 
? ?
E__inference_JointModel_layer_call_and_return_conditional_losses_76360p345678??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0????????? 
? ?
E__inference_JointModel_layer_call_and_return_conditional_losses_76388p345678??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0????????? 
? ?
*__inference_JointModel_layer_call_fn_75538d345678@?=
6?3
)?&
input_1?????????
p 

 
? "?????????? ?
*__inference_JointModel_layer_call_fn_75651d345678@?=
6?3
)?&
input_1?????????
p

 
? "?????????? ?
*__inference_JointModel_layer_call_fn_76315c345678??<
5?2
(?%
inputs?????????
p 

 
? "?????????? ?
*__inference_JointModel_layer_call_fn_76332c345678??<
5?2
(?%
inputs?????????
p

 
? "?????????? ?
 __inference__wrapped_model_75438?345678"#()r?o
h?e
c?`
.?+
ImageA_Input?????????
.?+
ImageB_Input?????????
? "1?.
,
dense_3!?
dense_3??????????
F__inference_concatenate_layer_call_and_return_conditional_losses_76401?Z?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1????????? 
? "%?"
?
0?????????@
? ?
+__inference_concatenate_layer_call_fn_76394vZ?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1????????? 
? "??????????@?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_76501l567?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_1_layer_call_fn_76490_567?4
-?*
(?%
inputs?????????
? " ???????????
A__inference_conv2d_layer_call_and_return_conditional_losses_76481l347?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_conv2d_layer_call_fn_76470_347?4
-?*
(?%
inputs?????????
? " ???????????
B__inference_dense_1_layer_call_and_return_conditional_losses_76421\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_76410O/?,
%?"
 ?
inputs?????????@
? "???????????
B__inference_dense_2_layer_call_and_return_conditional_losses_76441\"#/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_2_layer_call_fn_76430O"#/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_3_layer_call_and_return_conditional_losses_76461\()/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_3_layer_call_fn_76450O()/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_76552]780?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? y
%__inference_dense_layer_call_fn_76541P780?-
&?#
!?
inputs??????????
? "?????????? ?
B__inference_flatten_layer_call_and_return_conditional_losses_76532a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? 
'__inference_flatten_layer_call_fn_76526T7?4
-?*
(?%
inputs?????????
? "????????????
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76516?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_76521h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
-__inference_max_pooling2d_layer_call_fn_76506?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
-__inference_max_pooling2d_layer_call_fn_76511[7?4
-?*
(?%
inputs?????????
? " ???????????
H__inference_siamese_model_layer_call_and_return_conditional_losses_76019?345678"#()z?w
p?m
c?`
.?+
ImageA_Input?????????
.?+
ImageB_Input?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_siamese_model_layer_call_and_return_conditional_losses_76060?345678"#()z?w
p?m
c?`
.?+
ImageA_Input?????????
.?+
ImageB_Input?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_siamese_model_layer_call_and_return_conditional_losses_76228?345678"#()r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_siamese_model_layer_call_and_return_conditional_losses_76298?345678"#()r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
-__inference_siamese_model_layer_call_fn_75810?345678"#()z?w
p?m
c?`
.?+
ImageA_Input?????????
.?+
ImageB_Input?????????
p 

 
? "???????????
-__inference_siamese_model_layer_call_fn_75978?345678"#()z?w
p?m
c?`
.?+
ImageA_Input?????????
.?+
ImageB_Input?????????
p

 
? "???????????
-__inference_siamese_model_layer_call_fn_76128?345678"#()r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p 

 
? "???????????
-__inference_siamese_model_layer_call_fn_76158?345678"#()r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p

 
? "???????????
#__inference_signature_wrapper_76098?345678"#()???
? 
???
>
ImageA_Input.?+
ImageA_Input?????????
>
ImageB_Input.?+
ImageB_Input?????????"1?.
,
dense_3!?
dense_3?????????