# Data

**[train/validation/test]_sequences.csv** - the target sequences of the RNA molecules.

`target_id` - (string) An arbitrary identifier. In train_sequences.csv, this is formatted as pdb_id_chain_id, where pdb_id is the id of the entry in the Protein Data Bank and chain_id is the chain id of the monomer in the pdb file.
`sequence` - (string) The RNA sequence. For test_sequences.csv, this is guaranteed to be a string of A, C, G, and U. For some train_sequences.csv, other characters may appear.
`temporal_cutoff` - (string) The date in yyyy-mm-dd format that the sequence was published. See Additional Notes.
`description` - (string) Details of the origins of the sequence. For a few targets, additional information on small molecule ligands bound to the RNA is included. You don't need to make predictions for these ligand coordinates.
`all_sequences` - (string) FASTA-formatted sequences of all molecular chains present in the experimentally solved structure. In a few cases this may include multiple copies of the target RNA (look for the word "Chains" in the header) and/or partners like other RNAs or proteins or DNA. You don't need to make predictions for all these molecules; if you do, just submit predictions for sequence. Some entries are blank.

**[train/validation]_labels.csv** - experimental structures.

`ID` - (string) that identifies the target_id and residue number, separated by _. Note: residue numbers use one-based indexing.
`resname` - (character) The RNA nucleotide ( A, C, G, or U) for the residue.
`resid` - (integer) residue number.
`x_1,y_1,z_1,x_2,y_2,z_2,…` - (float) Coordinates (in Angstroms) of the C1' atom for each experimental RNA structure. There is typically one structure for the RNA sequence, and `train_labels.csv` curates one structure for each training sequence. However, in some targets the experimental method has captured more than one conformation, and each will be used as a potential reference for scoring your predictions. `validation_labels.csv` has examples of targets with multiple reference structures (`x_2,y_2,z_2,` etc.).

**train_[sequences/labels].v2.csv** - extracted from the protein data bank with full text search for keyword RNA relaxed filter for unstructured RNAs based on pairwise C1' distances, where 20% of residues have to be close to some other residue that is over 4 bases apart.

**sample_submission.csv**

Same format as train_labels.csv but with five sets of coordinates for each of your five predicted structures (``x_1,y_1,z_1,x_2,y_2,z_2,…x_5,y_5,z_5``).
You must submit five sets of coordinates.