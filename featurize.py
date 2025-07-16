from dgllife import utils
from rdkit import Chem
from rdkit.Chem import HybridizationType
import torch
import rdkit
def featurize_atoms(mol):
    feats=[]
    for atom in mol.GetAtoms():
        feat=utils.atom_type_one_hot(atom,allowable_set=['C','F',]) #Atom type
        feat+=utils.atom_degree_one_hot(atom,allowable_set=[1,2,3,4])  #Number of covalent bonds
        feat+=utils.atom_hybridization_one_hot(atom,allowable_set=[HybridizationType.SP2, HybridizationType.SP3])  #Hybridization
        feat+=utils.atom_total_num_H_one_hot(atom,allowable_set=[0,1,2,3]) # Number of connected hydrogens
        feats.append(feat)
    return {'atom':torch.tensor(feats).float()}
def featurize_bonds(mol):
    feats=[]
    for bond in mol.GetBonds():
        feat=utils.bond_type_one_hot(bond,allowable_set=[Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,]) #BondType
        feat+=utils.bond_stereo_one_hot(bond,allowable_set=[ rdkit.Chem.rdchem.BondStereo.STEREONONE,                                               rdkit.Chem.rdchem.BondStereo.STEREOZ,rdkit.Chem.rdchem.BondStereo.STEREOE])# Stereo
        feats.append(feat)
        feats.append(feat)
    return {'bond':torch.tensor(feats).float()}