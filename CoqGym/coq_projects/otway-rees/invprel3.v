(* Contribution to the Coq Library   V6.3 (July 1999)                    *)
Require Import securite.

Lemma POinvprel3 :
 forall (l l0 : list C) (k k0 k1 k2 : K) (c c0 c1 c2 : C)
   (d d0 d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 d11 d12 d13 d14 d15 d16 d17 d18 d19
    d20 : D),
 inv0
   (ABSI (MBNaKab d7 d8 d9 k0) (MANbKabCaCb d4 d5 d6 k c c0)
      (MABNaNbKeyK d d0 d1 d2 d3) l) ->
 inv1
   (ABSI (MBNaKab d7 d8 d9 k0) (MANbKabCaCb d4 d5 d6 k c c0)
      (MABNaNbKeyK d d0 d1 d2 d3) l) ->
 invP
   (ABSI (MBNaKab d7 d8 d9 k0) (MANbKabCaCb d4 d5 d6 k c c0)
      (MABNaNbKeyK d d0 d1 d2 d3) l) ->
 rel3
   (ABSI (MBNaKab d7 d8 d9 k0) (MANbKabCaCb d4 d5 d6 k c c0)
      (MABNaNbKeyK d d0 d1 d2 d3) l)
   (ABSI (MBNaKab d18 d19 d20 k2) (MANbKabCaCb d15 d16 d17 k1 c1 c2)
      (MABNaNbKeyK d10 d11 d12 d13 d14) l0) ->
 invP
   (ABSI (MBNaKab d18 d19 d20 k2) (MANbKabCaCb d15 d16 d17 k1 c1 c2)
      (MABNaNbKeyK d10 d11 d12 d13 d14) l0).

Proof.
do 32 intro.
unfold inv0, invP, rel3 in |- *; intros know_c_c0_l Inv1 know_Kab and1.
elim know_c_c0_l; intros know_c_l know_c0_l.
elim and1; intros eq_l0 t1.
clear know_c_c0_l Inv1 and1 t1.
rewrite eq_l0.
unfold quint in |- *.
apply D2.
simpl in |- *.
repeat apply C2 || apply C3 || apply C4.
apply
 equivncomp
  with
    (Encrypt
       (quad (B2C (D2B d17)) (B2C (D2B d4)) (B2C (D2B d16)) (B2C (D2B Bid)))
       (KeyX Bid) :: l ++ rngDDKKeyABminusKab).
apply AlreadyIn; apply E0; apply EP0; assumption.
unfold quad in |- *.
repeat apply C2 || apply C3 || apply C4.
apply D1; assumption.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
discriminate.
Qed.