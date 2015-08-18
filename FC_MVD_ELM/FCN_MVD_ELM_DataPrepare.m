% Script for Fully-conncted MDeep-ELM for shape segmentation.

clear;
%depth image and label image data dir 
depthAndColorPath='Ant128data\';
addpath(depthAndColorPath);
% mesh category name
className='Ant';
allMeshNum=[81,83:93,95:99];
% test mesh number
testIndex=3;
Angle=19;  % angle number 
resolution=128; % image resolution
dataPreparation;



