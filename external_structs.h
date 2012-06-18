#pragma once

#include "consts.h"

typedef struct {
    char type [PDB_ATOM_ATOM_NAME_LEN+1];
    double x,y,z;
    int backbone;
} Atom;

typedef struct {
    char pdb_id[PDB_ATOM_RES_NO_LEN+2];
    char res_type[PDB_ATOM_RES_NAME_LEN+1];
    char res_type_short;
    char chain;
    int no_atoms;
    Atom atom[MAX_NO_ATOMS];
    Atom *Ca;
    int interface;
    int solvent_accessible;
    int belongs_to_helix;
    int belongs_to_strand;
    int alt_belongs_to_helix; /* we allow a couple of residues overlap in SSEs */
    int alt_belongs_to_strand;
} Residue;

typedef struct {} SSElement;

typedef struct {
    int length;
    Residue * sequence;
    int no_helices;
    SSElement *helix;
    int no_strands;
    SSElement *strand;
    int * sse_sequence;
} Protein;

typedef struct {
    ////////////////////
    // general
    int size;                // max number of mapped elements (SSEs or Ca, depending where we use the structure
    int matches;             // the number ofactually mapped SSEs
    double q[4]; /* rotation (quaternion)  -- the rotation which results in this particular map */
    double T[3]; /* translation (once we believe we know it)  -- the translation (for the Ca level)*/
    ////////////////////
    // referring to SSE-level map:
    int *x2y, *y2x;          // x2y: for each mapped pair (x,y), returns y, the index of the SSE in the structure Y,
                             // given x, the index of SSE in structure X  ( x2y[x] = y, e.g. x2y[3] = 7)
    int x2y_size, y2x_size;  // the size of the two maps above; ultimately the two should be equal
                             // in some general intermediate step that might not be the case, in the
                             // current implementation it always is
    double avg_length_mismatch; // average difference in the length of mapped SSEs
    double rmsd;             /* rmsd for the centers of matched SSEs */
    ////////////////////
    // "urchin" scoring
    double **cosine;          // table of angle cosines for all pairs (x,y) (all of them, not just the mapped ones)
    double **image;           // table of exp terms for all pairs (x,y) (all of them, not just the mapped ones)
    double F;                 // value of the  function F for this map
    double avg, avg_sq;       // refers to the avg and average square of F over the sapce of all rotations
                              // the input for the calulcation of the z-score
    double z_score;           // z-score for F (based on avg and avg_sq0
    double assigned_score;    // sum of the exp terms of but only for the  matched SSE pairs (x,y)
    ////////////////////
    // referring to Ca-level map:
    int *x2y_residue_level, *y2x_residue_level;  // the same as the x2y above, this time not on SSE, but on Ca level
    int x2y_residue_l_size, y2x_residue_l_size;
    int res_almt_length;    // length of the alignment on the Ca level
    double res_rmsd;        /* rmsd for the matched Ca atoms*/
    double aln_score;         // like the "assigned score" above, but for the Ca level
    double res_almt_score;/*not sure - cou;ld be I am duplicating aln_score */
    ////////////////////

    // complementary or sub-maps - never mind for now, just leave as is
    int *submatch_best;         // the best map which complements this one; don't worry about it right now
    double score_with_children; // this goes with the submatch above = never mind
    double compl_z_score;       // z-score for the submatch
    ///////////////////

    // file to which the corresponding pdb was written
    char filename[MEDSTRING];
} Map;
