(* This program is free software; you can redistribute it and/or      *)
(* modify it under the terms of the GNU Lesser General Public License *)
(* as published by the Free Software Foundation; either version 2.1   *)
(* of the License, or (at your option) any later version.             *)
(*                                                                    *)
(* This program is distributed in the hope that it will be useful,    *)
(* but WITHOUT ANY WARRANTY; without even the implied warranty of     *)
(* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      *)
(* GNU General Public License for more details.                       *)
(*                                                                    *)
(* You should have received a copy of the GNU Lesser General Public   *)
(* License along with this program; if not, write to the Free         *)
(* Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA *)
(* 02110-1301 USA                                                     *)


Require Export DistributedReferenceCounting.machine1.machine.
Require Export DistributedReferenceCounting.machine1.cardinal.
Require Export DistributedReferenceCounting.machine1.comm.

Unset Standard Proposition Elimination Names.

Section MESSAGE.

Lemma not_s1_s2 : forall s s1 s2 : Site, s1 <> s2 -> s1 <> s \/ s2 <> s.
Proof.
  intros.
  case (eq_site_dec s1 s).
  intro; rewrite e.
  rewrite e in H.
  right; auto.
  
  intro.
  left; auto.
Qed.


Lemma not_owner_inc :
 forall (c : Config) (s1 s2 : Site),
 legal c -> ~ In_queue Message (inc_dec owner) (bm c s1 s2).
Proof.
  intro; intro; intro.
  intro.
  elim H.
  simpl in |- *.
  intuition.
  
  simple induction t; intros; simpl in |- *.
  apply not_in_post.
  discriminate.
  auto.
  
  apply not_in_collect.
  auto.
  
  apply not_in_post.
  discriminate.
  
  apply not_in_collect.
  auto.
  
  apply not_in_post.
  discriminate.
  
  apply not_in_collect.
  auto.
  
  apply not_in_collect.
  auto.
  
  apply not_in_post.
  injection.
  intuition.
  
  apply not_in_collect.
  auto.
  
  apply not_in_post.
  discriminate.
  auto.
Qed.

(* as formulated by Jean *)

Lemma not_owner_inc2 :
 forall (c : Config) (s0 s1 s2 : Site),
 legal c -> In_queue Message (inc_dec s0) (bm c s1 s2) -> s0 <> owner.
Proof.
  intros.
  generalize (not_owner_inc c s1 s2 H).
  intro.
  generalize
   (equality_from_queue_membership Message eq_message_dec 
      (inc_dec owner) (inc_dec s0) (bm c s1 s2)).
  intros.
  generalize (H2 H0 H1).
  intro.
  case (eq_site_dec s0 owner).
  intro.
  rewrite e in H3.
  generalize H3.
  intuition.
  auto.
Qed.

(* As I need it in my proof *)

Lemma not_owner_inc3 :
 forall (c : Config) (s1 s2 : Site),
 legal c ->
 forall s0 : Site, In_queue Message (inc_dec s0) (bm c s1 s2) -> s0 <> owner.
Proof.
  intros.
  apply (not_owner_inc2 c s0 s1 s2).
  auto.
  auto.
Qed.

Lemma not_owner_inc4 :
 forall (c : Config) (s1 s2 : Site),
 legal c ->
 forall (m : Message) (s0 : Site),
 m = inc_dec s0 -> In_queue Message m (bm c s1 s2) -> s0 <> owner.
Proof.
  intros.
  apply (not_owner_inc2 c s0 s1 s2).
  auto.
  rewrite H0 in H1.
  auto.
Qed.


(* The proof is long and very boring, but it is a simple case
   analysis.  Could it be possible to simplify it? *)

Lemma no_copy_to_owner :
 forall c0 : Config,
 legal c0 -> forall s1 : Site, ~ In_queue Message copy (bm c0 s1 owner).
Proof.
  intros c0 H.
  elim H.
  intros.
  simpl in |- *.
  intuition.
  
  simple induction t; intros; simpl in |- *.

  (* 1 *)

  rewrite post_elsewhere.
  auto.
  right; auto.

  (* 2 *)

  case (eq_queue_dec s1 s0 s2 owner).
  intro.
  decompose [and] a.
  rewrite H2; rewrite H3.
  rewrite collect_here.
  apply not_in_q_output.
  auto.
  intro; rewrite collect_elsewhere.
  auto.
  auto.

  (* 3 *)
  
  case (eq_queue_dec owner s3 s0 owner).
  intro.
  decompose [and] a.
  rewrite H3; rewrite <- H2.
  rewrite post_here.
  case (eq_site_dec s1 owner).
  intro.
  rewrite e0.
  rewrite collect_here.
  simpl in |- *.
  generalize (H1 owner).
  intro.
  generalize (not_in_q_output Message copy (bm c owner owner)).
  intro.
  generalize (H5 H4).
  intro.
  unfold not in |- *.
  intuition.
  discriminate.
  intro.
  rewrite collect_elsewhere.
  simpl in |- *.
  generalize (H1 owner).
  intuition.
  discriminate.
  left; auto.
  intro.
  rewrite post_elsewhere.
  case (eq_site_dec s1 s0).
  intro.
  rewrite e0.
  rewrite collect_here.
  apply not_in_q_output.
  auto.
  intro.
  rewrite collect_elsewhere.
  auto.
  auto.
  decompose [or] o.
  right; auto.
  left; auto.

  (* 4 *)
  
  case (eq_queue_dec s2 s0 s1 owner).
  intro.
  decompose [and] a.
  rewrite H2; rewrite H3.
  rewrite post_here.
  case (eq_site_dec s0 owner).
  intro.
  rewrite e1; rewrite collect_here.
  simpl in |- *.
  generalize (not_in_q_output Message copy (bm c owner owner)).
  intro.
  generalize (H1 owner).
  intuition.
  discriminate.
  intro.
  rewrite collect_elsewhere.
  simpl in |- *.
  generalize (H1 s0).
  intuition.
  discriminate.
  right; auto.
  intro.
  rewrite post_elsewhere.
  case (eq_queue_dec s1 s0 s2 owner).
  intro.
  decompose [and] a.
  rewrite H2; rewrite H3; rewrite collect_here.
  apply not_in_q_output.
  auto.
  intro.
  rewrite collect_elsewhere.
  auto.
  auto.
  auto.

  (* 5 *)
  
  case (eq_queue_dec owner s1 s2 owner).
  intro.
  decompose [and] a.
  rewrite <- H2; rewrite H3.
  rewrite collect_here.
  apply not_in_q_output.
  auto.
  intro.
  rewrite collect_elsewhere.
  auto.
  auto.

  (* 6 *)
  
  case (eq_site_dec s2 s0).
  intro.
  rewrite e1.
  rewrite post_here.
  case (eq_queue_dec s1 s0 s0 owner).
  intro.
  decompose [and] a.
  rewrite H2; rewrite <- H3.
  rewrite collect_here.
  simpl in |- *.
  generalize (not_in_q_output Message copy (bm c s0 s0)).
  intros.
  generalize (H1 s0).
  rewrite H3.
  rewrite H3 in H4.
  intro.
  generalize (H4 H5).
  intuition.
  discriminate.
  intro.
  rewrite collect_elsewhere.
  simpl in |- *.
  generalize (H1 s0).
  intro.
  intuition.
  discriminate.
  discriminate.
  auto.
  intro.
  rewrite post_elsewhere.
  case (eq_queue_dec s1 s0 s2 owner).
  intros.
  decompose [and] a.
  rewrite H2; rewrite H3.
  rewrite collect_here.
  apply not_in_q_output.
  auto.
  intro.
  rewrite collect_elsewhere.
  auto.
  auto.
  left; auto.

  (* 7 *)
  
  case (eq_site_dec s s1).
  intro.
  rewrite e1.
  rewrite post_here.
  simpl in |- *.
  generalize (H1 s1).
  intuition.
  discriminate.
  intro.
  rewrite post_elsewhere.
  auto.
  auto.

Qed.


Lemma empty_q_to_me :
 forall c : Config, legal c -> forall s : Site, bm c s s = empty Message.
Proof.
  intros c H.
  elim H.
  intros.
  simpl in |- *.
  unfold bag_init in |- *.
  auto.
  
  simple induction t; intros; simpl in |- *.

  (* 1 *)

  rewrite post_elsewhere; auto.
  apply not_s1_s2; auto.

  (* 2 *)
  
  case (eq_site_dec s1 s2).
  intro; generalize e.
  rewrite e0.
  rewrite H1.
  simpl in |- *; intro; discriminate.
  
  intros.
  rewrite collect_elsewhere; auto.
  apply not_s1_s2; auto.
 
  (* 3 *)
 
  generalize (not_owner_inc2 c0 s3 s1 owner H0).
  intros.
  rewrite post_elsewhere.
  case (eq_site_dec s1 owner).
  intro.
  generalize e.
  rewrite e0.
  rewrite H1.
  simpl in |- *; intro; discriminate.
  intros.
  rewrite collect_elsewhere; auto.
  apply not_s1_s2; auto.
  
  apply not_s1_s2; auto.
  cut (s3 <> owner).
  auto.
  
  apply H2.
  apply first_in.
  auto.

  (* 4 *)
  
  case (eq_site_dec s1 s2).
  intro; generalize e0.
  rewrite e1.
  rewrite H1.
  simpl in |- *.
  intro; discriminate.
  intro.
  rewrite post_elsewhere.
  rewrite collect_elsewhere.
  auto.
  apply not_s1_s2.
  auto.
  apply not_s1_s2.
  auto.

  (* 5 *)
  
  case (eq_site_dec owner s2).
  intro.
  generalize e0; rewrite e1; simpl in |- *.
  rewrite H1.
  simpl in |- *.
  intro; discriminate.
  
  intro.
  rewrite collect_elsewhere.
  auto.
  
  apply not_s1_s2.
  auto.

  (* 6 *)
  
  case (eq_site_dec s1 s2).
  intro; generalize e0.
  rewrite e1; simpl in |- *.
  rewrite H1.
  simpl in |- *; intro; discriminate.
  
  intro.
  generalize (no_copy_to_owner c0 H0 s1).
  intro.
  case (eq_site_dec s2 owner).
  intro.
  elim H2.
  apply first_in.
  rewrite <- e1; auto.
  
  intro.
  rewrite post_elsewhere.
  rewrite collect_elsewhere.
  auto.
  apply not_s1_s2.
  auto.
  apply not_s1_s2.
  auto.

  (* 7 *)
  
  rewrite post_elsewhere.
  auto.
  apply not_s1_s2.
  auto.
Qed.

Lemma not_reflexive :
 forall c : Config,
 legal c ->
 forall s1 s5 : Site,
 In_queue Message (inc_dec s1) (bm c s5 owner) -> s1 <> s5.
Proof.
  intros c H.
  elim H.
  simpl in |- *.
  intuition.
  intros c0 t.
  elim t.

  (* 1 *)

  simpl in |- *.
  intros.
  apply H1.
  apply (in_post Message (inc_dec s0) copy (bm c0) s1 s2).
  discriminate.
  auto.

  (* 2 *)
  
  simpl in |- *.
  intros.
  apply H1.
  apply (in_collect Message (inc_dec s0) (bm c0) s1 s2).
  auto.

  (* 3 *)
  
  simpl in |- *.
  intros.
  apply H1.
  apply (in_collect Message (inc_dec s0) (bm c0) s1 owner).
  generalize H2.
  simpl in |- *.
  case (eq_queue_dec owner s5 s3 owner).
  intro.
  decompose [and] a.
  rewrite H4; rewrite H3.
  rewrite post_here.
  simpl in |- *.
  intuition.
  discriminate.
  intro.
  rewrite post_elsewhere.
  auto.
  auto.

  (* 4 *)
  
  simpl in |- *.
  intros.
  apply H1.
  apply (in_collect Message (inc_dec s0) (bm c0) s1 s2).
  generalize H2.
  case (eq_queue_dec s2 s5 s1 owner).
  intro.
  decompose [and] a.
  rewrite H3; rewrite H4.
  rewrite post_here.
  simpl in |- *.
  intuition.
  discriminate.
  intro.
  rewrite post_elsewhere.
  auto.
  auto.

  (* 5 *)
  
  simpl in |- *.
  intros.
  apply H1.
  apply (in_collect Message (inc_dec s1) (bm c0) owner s2).
  auto.

  (* 6 *)
  
  simpl in |- *.
  intros.
  case (eq_site_dec s2 s5).
  intro.
  rewrite e1 in H2.
  generalize H2.
  rewrite post_here.
  simpl in |- *.
  case (eq_queue_dec s1 s5 s5 owner).
  intro.
  decompose [and] a.
  elim n.
  rewrite H3; auto.  
  intro.
  rewrite collect_elsewhere.
  generalize (H1 s0 s5).
  intro.
  case (eq_site_dec s0 s1).
  intro.
  intro.
  rewrite <- e1; rewrite e2.
  unfold not in |- *.
  intro.
  generalize e0.
  rewrite H5.
  rewrite empty_q_to_me.
  simpl in |- *.
  discriminate.  
  auto.  
  intros.
  apply H3.
  elim H4.
  intro.
  elim n0.
  inversion H5.
  auto.
  auto.
  auto.
  intro.
  generalize H2.
  rewrite post_elsewhere.
  intro.
  apply H1.
  apply (in_collect Message (inc_dec s0) (bm c0) s1 s2).
  auto.
  left; auto.

  (* 7 *)
  
  simpl in |- *.
  intros.
  apply H1.
  apply (in_post Message (inc_dec s1) dec (bm c0) s owner).
  discriminate.
  auto.
Qed.

Lemma not_in_queue_post :
 forall (b : Bag_of_message) (s1 s2 s3 s4 : Site) (m1 m2 : Message),
 m1 <> m2 ->
 ~ In_queue Message m1 (b s1 s2) ->
 ~ In_queue Message m1 (Post_message Message m2 b s3 s4 s1 s2).
Proof.
   intros.
   case (eq_queue_dec s3 s1 s4 s2).
   intro; decompose [and] a.
   rewrite H1; rewrite H2; rewrite post_here; simpl in |- *.
   intuition.
   intros; rewrite post_elsewhere; auto.
Qed.

           
Lemma not_in_queue_collect :
 forall (b : Bag_of_message) (s1 s2 s3 s4 : Site) (m1 : Message),
 ~ In_queue Message m1 (b s1 s2) ->
 ~ In_queue Message m1 (Collect_message Message b s3 s4 s1 s2).
Proof.
   intros.
   case (eq_queue_dec s3 s1 s4 s2).
   intro; decompose [and] a.
   rewrite H0; rewrite H1; rewrite collect_here.
   apply not_in_q_output.
   auto.
   intro; rewrite collect_elsewhere; auto.
Qed.


Lemma inc_dec_owner :
 forall (c : Config) (s1 s2 : Site),
 legal c ->
 s1 = owner \/ s2 <> owner ->
 forall s0 : Site, ~ In_queue Message (inc_dec s0) (bm c s1 s2).
Proof.
  intros c s1 s2 H.
  elim H.
  simpl in |- *.
  intros; unfold not in |- *; auto.
  
  simple induction t; simpl in |- *; intros.
  apply not_in_queue_post.
  discriminate.
  
  auto.
  
  apply not_in_queue_collect.
  auto.
  
  apply not_in_queue_post.
  discriminate.
  
  apply not_in_queue_collect; auto.
  
  apply not_in_queue_post.
  discriminate.
  
  apply not_in_queue_collect.
  auto.
  
  apply not_in_queue_collect.
  auto.
  
  case (eq_queue_dec s3 s1 owner s2).
  intro; decompose [and] a.
  elim H2.
  intro.
  cut (s3 <> owner).
  intro.
  elim H6.
  rewrite H3; rewrite H5; auto.
  
  unfold not in |- *; intro.
  generalize (no_copy_to_owner c0 H0 s0).
  intro.
  elim H7; rewrite <- H6; auto.
  apply first_in.
  auto.
  
  intro.
  elim H3; auto.
  
  intro; rewrite post_elsewhere.
  apply not_in_queue_collect.
  auto.
  
  auto.
  
  apply not_in_queue_post.
  discriminate.
  
  auto.
    
Qed.


Lemma inc_dec_owner2 :
 forall (c : Config) (s1 s2 : Site),
 legal c ->
 s1 <> owner ->
 s2 <> owner ->
 forall s0 : Site, ~ In_queue Message (inc_dec s0) (bm c s1 s2).
Proof.
intros.
apply inc_dec_owner; auto.
Qed.

Lemma inc_dec_owner3 :
 forall (c : Config) (s1 s2 : Site),
 legal c ->
 s2 <> owner ->
 forall s0 : Site, ~ In_queue Message (inc_dec s0) (bm c s1 s2).
Proof.
intros.
apply inc_dec_owner; auto.
Qed.


Lemma inc_dec_owner4 :
 forall (c : Config) (s1 s2 : Site),
 legal c ->
 s1 = owner -> forall s0 : Site, ~ In_queue Message (inc_dec s0) (bm c s1 s2).
Proof.
intros.
apply inc_dec_owner; auto.
Qed.

Lemma empty_queue1 :
 forall c : Config,
 legal c -> forall s1 s2 : Site, s1 = s2 -> bm c s1 s2 = empty Message.
Proof.
intros.
generalize (empty_q_to_me c H s1).
rewrite H0.
auto.
Qed.

Lemma empty_queue2 :
 forall c : Config,
 legal c ->
 forall s1 s2 s : Site, s1 = s -> s2 = s -> bm c s1 s2 = empty Message.
Proof.
  intros.
  generalize (empty_queue1 c H s1 s2).
  rewrite H0.
  rewrite H1.
  auto.
Qed.

Lemma st_rt :
 forall (c0 : Config) (s0 : Site),
 legal c0 -> s0 <> owner -> (st c0 s0 > 0)%Z -> rt c0 s0 = true.
Proof.
  intros c0 s0 H.
  elim H.
  simpl in |- *.
  unfold send_init in |- *.
  intros.
  cut (~ (0 > 0)%Z).
  intro.
  elim H2; auto.
  omega.

  intros c t.
  elim t.

  (* 1 *)

  simpl in |- *.
  intros s1 s2.
  case (eq_site_dec s0 s1); intro.
  unfold access in |- *.
  rewrite e.
  intros.
  elim a.
  intro; elim H2; auto.
  auto.
  intros.
  apply H1.
  auto.
  generalize H3; unfold Inc_send_table in |- *.
  rewrite other_site.
  auto.
  auto.


  (* 2 *)  

  simpl in |- *.
  intros.
  apply H1.
  auto.
  generalize H3; unfold Dec_send_table in |- *.
  case (eq_site_dec s0 s2).
  intro.
  rewrite e0; rewrite that_site.
  intro.
  omega.
  intro; rewrite other_site; auto.

  (* 3 *)
  
  simpl in |- *.
  intros.
  apply H1.
  auto.
  generalize H3; unfold Inc_send_table in |- *; rewrite other_site; auto.
  
  (* 4 *)

  simpl in |- *.
  intros.
  apply H1; auto.

  (* 5 *)
  
  simpl in |- *.
  intros.
  case (eq_site_dec s2 s0).
  intro; unfold Set_rec_table in |- *.
  rewrite e1; rewrite that_site; auto.
  intro.
  unfold Set_rec_table in |- *; rewrite other_site.
  apply H1; auto.
  auto.

  (* 6 *)
  
  simpl in |- *.
  intros.
  case (eq_site_dec s2 s0).
  intro; unfold Set_rec_table in |- *.
  rewrite e1; rewrite that_site; auto.
  intro; unfold Set_rec_table in |- *; rewrite other_site.
  apply H1; auto.
  auto.

  (* 7 *)
  
  simpl in |- *.
  intros.
  case (eq_site_dec s s0).
  intro.
  rewrite <- e1 in H3.
  rewrite e in H3.
  absurd (0 > 0)%Z.
  omega.
  auto.
  intro; unfold Reset_rec_table in |- *.
  rewrite other_site.
  apply H1; auto.
  auto.
Qed.



End MESSAGE.
