#
# encoding: latin-1
#
# Code to implement Radial Basis Function (RBF) neural networks
#
# Pedro Frazão Ferreira, 2009
#
# This code is a property of the Centre for Intelligent Systems and the 
# Algarve Science and Technology Park.
# It may only be used by others after writen permission has been granted.
#
# Copyright, 2009:
#   Centre for Intelligent Systems
#   University of Algarve
#   Faro, Portugal
#
#   Algarve Science and Technology Park
#   Faro, Portugal
#

###################################################################################

#from numpy import dot, empty
import numpy
from rbfcore import comp_act_gauss

#####################################################################################################

class rbf:
  """
  Class to represent an RBF neural network.

   input_data,tti: Each line is an input pattern
  output_data,tto: Single column of corresponding output values

  C : network centres, one per line
  """
  def __init__(self,train_data,test_data=None,neurons=0,emptyn=False,bias=False,afun='gauss',iC='oakm'):
    self.afuns={'gauss':comp_act_gauss}
    if not emptyn:
      self.bias=bias
      n=neurons
      if self.bias:
        n+=1
      self.test=None
      self.neurons=neurons
      self.C=empty([])
      self.S=array([])
      self.W=array([])
      self.afun=afun
      self.iCentres=iC
      self.ActFunction=self.afuns[afun]
      self.state=None

####################################################################

  def ProduceOutput(self,input):
    n=self.neurons
    if self.bias:
      n+=1
    A=empty((input.shape[0],n))
    if self.bias:
      A[:,-1]=1.0
    self.ActFunction(input,self.C,self.S,A,array([]))
    return dot(A,self.W[0])




