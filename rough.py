s = """O
B_amino_acid_monomer
I_amino_acid_monomer
B_peptide
I_peptide
B_protein_N/A
I_protein_N/A
B_protein_complex
I_protein_complex
B_protein_domain_or_region
I_protein_domain_or_region
B_protein_family_or_group
I_protein_family_or_group
B_protein_molecule
I_protein_molecule
B_protein_substructure
I_protein_substructure
B_protein_subunit
I_protein_subunit
B_nucleotide
I_nucleotide
B_polynucleotide
I_polynucleotide
B_DNA_N/A
I_DNA_N/A
B_DNA_domain_or_region
I_DNA_domain_or_region
B_DNA_family_or_group
I_DNA_family_or_group
B_DNA_molecule
I_DNA_molecule
B_DNA_substructure
I_DNA_substructure
B_RNA_N/A
I_RNA_N/A
B_RNA_domain_or_region
I_RNA_domain_or_region
B_RNA_family_or_group
I_RNA_family_or_group
B_RNA_molecule
I_RNA_molecule
B_RNA_substructure
I_RNA_substructure
B_other_organic_compound
I_other_organic_compound
B_organic
I_organic
B_inorganic
I_inorganic
B_atom
I_atom
B_carbohydrate
I_carbohydrate
B_lipid
I_lipid
B_virus
I_virus
B_mono_cell
I_mono_cell
B_multi_cell
I_multi_cell
B_body_part
I_body_part
B_tissue
I_tissue
B_cell_type
I_cell_type
B_cell_component
I_cell_component
B_cell_line
I_cell_line
B_other_artificial_source
I_other_artificial_source
B_other_name
I_other_name"""

x = list(s.split("\n"))
print(len(x))