//! Rigid clusters, the rotatable-bond tree, and forward kinematics.
//!
//! A ligand conformation is fully described by
//!
//! * a **rigid-body transform** — a translation `t` and a rotation vector `θ`
//!   (the exponential map of SO(3), so the parameterisation is minimal and
//!   smooth at the origin), and
//! * one **torsion angle** per rotatable bond in the tree.
//!
//! Atoms are grouped into **rigid clusters** by cutting every rotatable bond:
//! two atoms that remain connected by a path of non-rotatable bonds always move
//! together. The clusters form a tree rooted at the largest one, and the
//! torsion on the bond that joins a cluster to its parent rotates the whole
//! subtree about that bond's axis.
//!
//! # Provenance
//!
//! The tree decomposition and the definition of a rotatable bond follow the
//! PDBQT preparation rules published with AutoDock 4.2 (Morris et al., *J.
//! Comput. Chem.* **29**, 2789, 2008; GPL-2.0) and Meeko (Morris et al.,
//! *PLoS ONE* **17**, e0163573, 2022; LGPL-2.1). The SO(3) exponential map and
//! its right Jacobian are standard textbook robotics (Barfoot, *State
//! Estimation for Robotics*, 2017) and are implemented here from first
//! principles. No AutoDockTools code was consulted.

use nalgebra::{Matrix3, Vector3};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error::{DockError, Result};
use crate::types::{dist3, Element, Molecule, Vec3};

/// Convert a `Vec3` alias into an `nalgebra` vector.
#[inline]
pub fn v3(a: Vec3) -> Vector3<f64> {
    Vector3::new(a[0], a[1], a[2])
}

/// Convert an `nalgebra` vector back into the `Vec3` alias.
#[inline]
pub fn to_v3(a: &Vector3<f64>) -> Vec3 {
    [a[0], a[1], a[2]]
}

/// Axis-angle (rotation vector) to rotation matrix, via Rodrigues' formula.
///
/// `θ` is the angle in radians; its direction is the rotation axis.
pub fn rotvec_to_matrix(theta: &Vector3<f64>) -> Matrix3<f64> {
    let angle = theta.norm();
    if angle < 1e-12 {
        // First-order expansion: R ≈ I + [θ]×
        let k = skew(theta);
        return Matrix3::identity() + k;
    }
    let axis = theta / angle;
    let k = skew(&axis);
    let s = angle.sin();
    let c = angle.cos();
    Matrix3::identity() + s * k + (1.0 - c) * (k * k)
}

/// The skew-symmetric matrix `[v]×` such that `[v]× w = v × w`.
#[inline]
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v[2], v[1], //
        v[2], 0.0, -v[0], //
        -v[1], v[0], 0.0,
    )
}

/// The right Jacobian of SO(3) at the rotation vector `θ`.
///
/// For `R(θ) = exp([θ]×)` the differential satisfies
///
/// ```text
/// dR = R · [J_r(θ) dθ]×
/// ```
///
/// so `J_r` maps a body-frame angular rate into the world frame. Its closed
/// form is
///
/// ```text
/// J_r(θ) = I − ((1−cos|θ|)/|θ|²)[θ]× + ((|θ|−sin|θ|)/|θ|³)[θ]×²
/// ```
///
/// and its small-angle series is `I − ½[θ]× + ⅙[θ]×² + …`.
pub fn so3_right_jacobian(theta: &Vector3<f64>) -> Matrix3<f64> {
    let k = skew(theta);
    let angle = theta.norm();
    if angle < 1e-6 {
        // Taylor expansion; the leading skew term carries a minus sign.
        return Matrix3::identity() - 0.5 * k + (1.0 / 6.0) * (k * k);
    }
    let a = (1.0 - angle.cos()) / (angle * angle);
    let b = (angle - angle.sin()) / (angle * angle * angle);
    Matrix3::identity() - a * k + b * (k * k)
}

/// A unit quaternion `[w, x, y, z]`, used by the Python bindings and I/O.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Quaternion {
    /// Scalar part.
    pub w: f64,
    /// `x` component.
    pub x: f64,
    /// `y` component.
    pub y: f64,
    /// `z` component.
    pub z: f64,
}

impl Quaternion {
    /// The identity rotation.
    pub const IDENTITY: Quaternion = Quaternion {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Build a quaternion from a rotation vector.
    pub fn from_rotvec(theta: &Vector3<f64>) -> Quaternion {
        let angle = theta.norm();
        if angle < 1e-12 {
            return Quaternion::IDENTITY;
        }
        let s = (angle * 0.5).sin();
        let v = theta / angle;
        Quaternion {
            w: (angle * 0.5).cos(),
            x: v[0] * s,
            y: v[1] * s,
            z: v[2] * s,
        }
    }

    /// Convert back to a rotation vector (axis × angle, with the angle in
    /// `(-π, π]`).
    pub fn to_rotvec(self) -> Vector3<f64> {
        let n = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if n < 1e-12 {
            return Vector3::zeros();
        }
        // angle = 2·atan2(|v|, w). Using atan2(w, |v|) here would return the
        // complementary angle and silently scale the vector.
        let mut angle = 2.0 * n.atan2(self.w);
        // Canonicalise to the shorter arc.
        if angle > std::f64::consts::PI {
            angle -= 2.0 * std::f64::consts::PI;
        }
        let s = angle / n;
        Vector3::new(self.x * s, self.y * s, self.z * s)
    }

    /// Convert to a 3×3 rotation matrix.
    pub fn to_matrix(self) -> Matrix3<f64> {
        rotvec_to_matrix(&self.to_rotvec())
    }

    /// Normalise, flipping the sign so that `w >= 0` (canonical hemisphere).
    pub fn normalized(self) -> Quaternion {
        let n = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if n < 1e-12 {
            return Quaternion::IDENTITY;
        }
        let mut q = Quaternion {
            w: self.w / n,
            x: self.x / n,
            y: self.y / n,
            z: self.z / n,
        };
        if q.w < 0.0 {
            q = Quaternion {
                w: -q.w,
                x: -q.x,
                y: -q.y,
                z: -q.z,
            };
        }
        q
    }
}

/// A group of atoms that always move together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RigidCluster {
    /// Indices of the atoms belonging to this cluster, ascending.
    pub atoms: Vec<usize>,
    /// Index of the parent cluster, or `None` for the root.
    pub parent: Option<usize>,
    /// Indices of the child clusters.
    pub children: Vec<usize>,
    /// Index of the torsion that joins this cluster to its parent.
    pub torsion: Option<usize>,
    /// Reference-frame origin of the cluster: the position of [`Self::axis_distal`]
    /// (the root cluster uses the coordinate origin).
    pub origin: Vec3,
    /// The atom at the proximal end of the parent torsion axis, in the root frame.
    pub axis_proximal: Option<usize>,
    /// The atom at the distal end of the parent torsion axis, in the root frame.
    pub axis_distal: Option<usize>,
}

/// A rotatable bond and the two clusters it separates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Torsion {
    /// Index of the atom on the proximal (parent) side.
    pub proximal_atom: usize,
    /// Index of the atom on the distal (child) side.
    pub distal_atom: usize,
    /// Cluster on the proximal side.
    pub proximal_cluster: usize,
    /// Cluster on the distal side.
    pub distal_cluster: usize,
}

/// A complete conformer: rigid-body transform plus one angle per torsion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Conformation {
    /// Translation applied to the reference frame, in Ångström.
    pub position: Vec3,
    /// Rotation vector, in radians (axis × angle).
    pub orientation: Vec3,
    /// Torsion angles, in radians, one per [`KinematicTree::torsions`] entry.
    pub torsions: Vec<f64>,
}

impl Conformation {
    /// A conformation that reproduces the reference coordinates exactly.
    pub fn identity(num_torsions: usize) -> Conformation {
        Conformation {
            position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0],
            torsions: vec![0.0; num_torsions],
        }
    }

    /// Number of degrees of freedom: 6 rigid-body + one per torsion.
    pub fn ndof(&self) -> usize {
        6 + self.torsions.len()
    }

    /// Pack into a flat gradient-sized vector.
    pub fn as_slice(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(6 + self.torsions.len());
        v.extend_from_slice(&self.position);
        v.extend_from_slice(&self.orientation);
        v.extend_from_slice(&self.torsions);
        v
    }

    /// Unpack from a flat vector produced by [`Self::as_slice`].
    pub fn from_slice(s: &[f64]) -> Result<Conformation> {
        if s.len() < 6 {
            return Err(DockError::molecule("conformation needs at least 6 values"));
        }
        Ok(Conformation {
            position: [s[0], s[1], s[2]],
            orientation: [s[3], s[4], s[5]],
            torsions: s[6..].to_vec(),
        })
    }
}

/// The rotatable-bond tree of a ligand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KinematicTree {
    /// Rigid clusters, topologically ordered (a parent always precedes its children).
    pub clusters: Vec<RigidCluster>,
    /// Rotatable bonds.
    pub torsions: Vec<Torsion>,
    /// Index of the root cluster in [`Self::clusters`].
    pub root: usize,
    /// For each atom, the cluster it belongs to.
    pub atom_cluster: Vec<usize>,
    /// Reference coordinates, copied from the input molecule.
    pub reference: Vec<Vec3>,
    /// Per-cluster cached reference-frame offsets, `ref[i] - cluster.origin`.
    offsets: Vec<Vec<(usize, [f64; 3])>>,
    /// For each torsion, the normalised axis direction in the reference frame.
    /// The global axis at search time is `rot_global[parent] * axis_dirs[t]`.
    axis_dirs: Vec<[f64; 3]>,
}

impl KinematicTree {
    /// Build the tree for `mol`.
    ///
    /// `declared_torsions` lists `(proximal, distal)` atom-index pairs that the
    /// input file asserted (from `TORSION` records). When non-empty these are
    /// used verbatim, so a prepared ligand docks with exactly the torsions its
    /// preparer intended. Otherwise torsions are perceived from the graph.
    pub fn from_molecule(
        mol: &Molecule,
        declared_torsions: &[(usize, usize)],
    ) -> Result<KinematicTree> {
        if mol.is_empty() {
            return Err(DockError::molecule(
                "cannot build a kinematic tree for an empty molecule",
            ));
        }
        let n = mol.atoms.len();
        let mut rot = if declared_torsions.is_empty() {
            perceive_rotatable_bonds(mol)
        } else {
            let mut v = Vec::new();
            for &(a, b) in declared_torsions {
                let key = (a.min(b), a.max(b));
                if a < n && b < n && mol.neighbors[a].contains(&b) && !v.contains(&key) {
                    v.push(key);
                }
            }
            // Restore the caller's order (proximal → distal matters for the tree).
            let mut ordered = Vec::new();
            for &(a, b) in declared_torsions {
                if a < n && b < n && mol.neighbors[a].contains(&b) && !ordered.contains(&(a, b)) {
                    ordered.push((a, b));
                }
            }
            if ordered.is_empty() {
                v
            } else {
                ordered
            }
        };
        rot.sort_unstable();
        rot.dedup();

        // A bond can only rotate if cutting it really separates the molecule.
        // `rot` is a candidate set, and a candidate whose two ends are already
        // connected through other bonds lies on a cycle — a ring bond that was
        // mislabelled, or a bond listed twice. Contracting those and repeating
        // converges, because every round unions at least one more pair.
        //
        // Without this the cluster graph could be cyclic, in which case the BFS
        // below cannot give every candidate its own child cluster and the
        // tree would be built from a mismatched torsion vector.
        let is_torsion = |rot: &[(usize, usize)], bond: &crate::types::Bond| {
            rot.binary_search(&(bond.i.min(bond.j), bond.i.max(bond.j)))
                .is_ok()
        };
        let mut uf = UnionFind::new(n);
        loop {
            for bond in &mol.bonds {
                if !is_torsion(&rot, bond) {
                    uf.union(bond.i, bond.j);
                }
            }
            let redundant: Vec<(usize, usize)> = rot
                .iter()
                .copied()
                .filter(|&(a, b)| uf.find(a) == uf.find(b))
                .collect();
            if redundant.is_empty() {
                break;
            }
            // Rebuild the union-find so the removals below take effect.
            let mut next_uf = UnionFind::new(n);
            for bond in &mol.bonds {
                if !is_torsion(&rot, bond) {
                    next_uf.union(bond.i, bond.j);
                }
            }
            for &(a, b) in &redundant {
                log::debug!("torsion {a}-{b} lies on a cycle; contracting it into a rigid cluster");
                next_uf.union(a, b);
            }
            uf = next_uf;
            rot.retain(|&(a, b)| uf.find(a) != uf.find(b));
        }

        // Union-find over the non-rotatable bonds gives the rigid clusters.
        let mut uf = UnionFind::new(n);
        for bond in &mol.bonds {
            if !is_torsion(&rot, bond) {
                uf.union(bond.i, bond.j);
            }
        }

        // Collect clusters, keeping the representative's atom ordering stable.
        let mut cluster_of: Vec<Option<usize>> = vec![None; n];
        let mut clusters: Vec<RigidCluster> = Vec::new();
        for atom in 0..n {
            let root = uf.find(atom);
            let cid = match cluster_of[root] {
                Some(c) => c,
                None => {
                    let c = clusters.len();
                    cluster_of[root] = Some(c);
                    clusters.push(RigidCluster {
                        atoms: Vec::new(),
                        parent: None,
                        children: Vec::new(),
                        torsion: None,
                        origin: [0.0, 0.0, 0.0],
                        axis_proximal: None,
                        axis_distal: None,
                    });
                    c
                }
            };
            clusters[cid].atoms.push(atom);
        }

        // Root = the largest cluster; ties break on the lowest atom index, which
        // makes the choice deterministic regardless of input ordering.
        let root = (0..clusters.len())
            .max_by_key(|&c| (clusters[c].atoms.len(), usize::MAX - clusters[c].atoms[0]))
            .unwrap_or(0);
        for c in 0..clusters.len() {
            clusters[c].atoms.sort_unstable();
        }

        // Build the torsion list, then topologically order the clusters via BFS.
        let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); clusters.len()];
        let mut torsions: Vec<Torsion> = Vec::new();
        for &(a, b) in &rot {
            let ca = cluster_of[uf.find(a)].expect("cluster assigned");
            let cb = cluster_of[uf.find(b)].expect("cluster assigned");
            if ca == cb {
                continue; // degenerate: the cut did not actually separate
            }
            let t = Torsion {
                proximal_atom: a,
                distal_atom: b,
                proximal_cluster: ca,
                distal_cluster: cb,
            };
            adj[ca].push((cb, torsions.len()));
            adj[cb].push((ca, torsions.len()));
            torsions.push(t);
        }

        // BFS from the root to establish parents and a topological order.
        let mut visited = vec![false; clusters.len()];
        visited[root] = true;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(root);
        let mut order = vec![root];
        while let Some(c) = queue.pop_front() {
            for &(nb, ti) in &adj[c] {
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;
                clusters[nb].parent = Some(c);
                clusters[nb].torsion = Some(ti);
                clusters[c].children.push(nb);
                // Orient the torsion proximal → distal relative to the parent.
                let mut t = torsions[ti];
                if t.proximal_cluster != c {
                    std::mem::swap(&mut t.proximal_atom, &mut t.distal_atom);
                    std::mem::swap(&mut t.proximal_cluster, &mut t.distal_cluster);
                }
                t.proximal_cluster = c;
                t.distal_cluster = nb;
                torsions[ti] = t;
                order.push(nb);
                queue.push_back(nb);
            }
        }

        // Any cluster not reached (a disconnected input) is attached to the root
        // as a torsion-free child so that every atom still moves.
        for c in 0..clusters.len() {
            if !visited[c] {
                visited[c] = true;
                clusters[c].parent = Some(root);
                clusters[c].torsion = None;
                clusters[root].children.push(c);
                order.push(c);
            }
        }

        // Re-index clusters into topological order.
        let mut remap = vec![usize::MAX; clusters.len()];
        for (new_idx, &old) in order.iter().enumerate() {
            remap[old] = new_idx;
        }
        let mut new_clusters: Vec<RigidCluster> = Vec::with_capacity(order.len());
        for &old in &order {
            let mut cl = clusters[old].clone();
            cl.parent = cl.parent.map(|p| remap[p]);
            for ch in &mut cl.children {
                *ch = remap[*ch];
            }
            new_clusters.push(cl);
        }
        for t in &mut torsions {
            t.proximal_cluster = remap[t.proximal_cluster];
            t.distal_cluster = remap[t.distal_cluster];
        }

        // Renumber the torsion vector so that torsion index order matches
        // cluster order, which is what `Conformation::torsions` follows.
        //
        // The contraction loop above guarantees a cluster graph that is a
        // forest, so every torsion is claimed here by exactly one child
        // cluster. If it somehow is not, report it as malformed input rather
        // than panicking: this runs inside a `panic = "abort"` binary reached
        // from Python, where a panic tears down the caller's process.
        let mut sorted_torsions: Vec<Option<Torsion>> = vec![None; torsions.len()];
        let mut next = 0usize;
        for cl in &mut new_clusters {
            if let Some(old_t) = cl.torsion {
                if next >= sorted_torsions.len() {
                    return Err(DockError::molecule(
                        "kinematic tree has more child clusters than rotatable bonds",
                    ));
                }
                cl.torsion = Some(next);
                sorted_torsions[next] = Some(torsions[old_t]);
                next += 1;
            }
        }
        let mut torsions: Vec<Torsion> = Vec::with_capacity(next);
        for t in sorted_torsions.into_iter().take(next) {
            match t {
                Some(t) => torsions.push(t),
                None => {
                    return Err(DockError::molecule(
                        "a rotatable bond was not claimed by exactly one child cluster",
                    ))
                }
            }
        }

        // Set cluster origins and axes now that the tree is final.
        //
        // A cluster with no torsion is a disconnected component that was
        // attached to the root rigidly; it rides along with the root frame and
        // needs no axis of its own.
        let reference: Vec<Vec3> = mol.atoms.iter().map(|a| a.coord).collect();
        for (ci, cl) in new_clusters.iter_mut().enumerate() {
            if ci == 0 {
                cl.origin = [0.0, 0.0, 0.0];
            } else if let Some(t) = cl.torsion {
                cl.axis_proximal = Some(torsions[t].proximal_atom);
                cl.axis_distal = Some(torsions[t].distal_atom);
                cl.origin = reference[torsions[t].distal_atom];
            } else {
                cl.origin = [0.0, 0.0, 0.0];
            }
        }

        let atom_cluster: Vec<usize> = (0..n)
            .map(|a| remap[cluster_of[uf.find(a)].expect("cluster assigned")])
            .collect();

        let offsets: Vec<Vec<(usize, [f64; 3])>> = new_clusters
            .iter()
            .map(|cl| {
                let o = v3(cl.origin);
                cl.atoms
                    .iter()
                    .map(|&a| (a, to_v3(&(v3(reference[a]) - o))))
                    .collect()
            })
            .collect();

        // Cache the normalised reference axis of every torsion. The global axis
        // at search time is just `rot_global[parent] * axis_dirs[t]`.
        let axis_dirs: Vec<[f64; 3]> = torsions
            .iter()
            .map(|t| {
                let d = v3(reference[t.distal_atom]) - v3(reference[t.proximal_atom]);
                let n = d.norm();
                if n < 1e-9 {
                    to_v3(&Vector3::z())
                } else {
                    to_v3(&(d / n))
                }
            })
            .collect();

        Ok(KinematicTree {
            clusters: new_clusters,
            torsions,
            root: 0,
            atom_cluster,
            reference,
            offsets,
            axis_dirs,
        })
    }
    /// Number of rotatable bonds.
    pub fn num_torsions(&self) -> usize {
        self.torsions.len()
    }

    /// Total number of optimisable degrees of freedom.
    pub fn ndof(&self) -> usize {
        6 + self.torsions.len()
    }

    /// Cluster frames in the **ligand-local frame**, i.e. after the torsion
    /// rotations but before the rigid-body transform.
    ///
    /// Keeping the two stages separate is what makes the rigid-body gradient
    /// well posed: the lever arm of a rigid rotation is exactly the
    /// torsion-posed position, so `∂r_i/∂θ` is `R · ∂m_i/∂θ` with a clean
    /// chain rule. Folding the torsion into world space instead would make the
    /// body and torsion rotations fail to commute.
    pub fn body_frames(&self, conf: &Conformation) -> (Vec<Vector3<f64>>, Vec<Matrix3<f64>>) {
        let n = self.clusters.len();
        let mut origin: Vec<Vector3<f64>> = self.clusters.iter().map(|c| v3(c.origin)).collect();
        let mut rot: Vec<Matrix3<f64>> = vec![Matrix3::identity(); n];

        for ci in 1..n {
            let cl = &self.clusters[ci];
            let parent = cl.parent.expect("non-root cluster has a parent");
            let o_parent = v3(self.clusters[parent].origin);
            let place = |local: Vector3<f64>| origin[parent] + rot[parent] * (local - o_parent);

            // A cluster with no torsion of its own is rigidly welded to its
            // parent (a disconnected component, or a bond the contraction step
            // absorbed). It simply inherits the parent frame — no rotation, and
            // its origin carried across unchanged.
            let ti = match cl.torsion {
                Some(t) => t,
                None => {
                    origin[ci] = place(v3(cl.origin));
                    rot[ci] = rot[parent];
                    continue;
                }
            };
            let pa = cl.axis_proximal.expect("axis");
            let da = cl.axis_distal.expect("axis");

            // The two axis atoms live in the *parent* cluster, so they must be
            // carried by the parent's already-accumulated frame.
            let pivot = place(v3(self.reference[da]));
            let axis_start = place(v3(self.reference[pa]));

            // Where this cluster's origin sits before its own torsion is applied.
            let base = place(v3(cl.origin));
            let direction = (pivot - axis_start).normalize();
            let angle = conf.torsions.get(ti).copied().unwrap_or(0.0);
            let local_rot = rotvec_to_matrix(&(direction * angle));

            rot[ci] = local_rot * rot[parent];
            origin[ci] = pivot + local_rot * (base - pivot);
        }
        (origin, rot)
    }

    /// Compute the global frame (origin + rotation) of every rigid cluster.
    ///
    /// This is the single source of truth for forward kinematics: [`Self::apply`],
    /// [`Self::conf_gradient`] and [`Self::to_global`] all route through it, so
    /// they cannot drift apart.
    pub fn global_frames(&self, conf: &Conformation) -> (Vec<Vector3<f64>>, Vec<Matrix3<f64>>) {
        let (body_origin, body_rot) = self.body_frames(conf);
        let r_body = rotvec_to_matrix(&v3(conf.orientation));
        let t_body = v3(conf.position);
        let origin_global: Vec<Vector3<f64>> =
            body_origin.iter().map(|o| t_body + r_body * o).collect();
        let rot_global: Vec<Matrix3<f64>> = body_rot.iter().map(|r| r_body * r).collect();
        (origin_global, rot_global)
    }

    /// Global position of one atom under `conf`.
    pub fn atom_position(
        &self,
        frames: &(Vec<Vector3<f64>>, Vec<Matrix3<f64>>),
        atom: usize,
    ) -> Vector3<f64> {
        let (origin_global, rot_global) = frames;
        let ci = self.atom_cluster[atom];
        origin_global[ci]
            + rot_global[ci] * (v3(self.reference[atom]) - v3(self.clusters[ci].origin))
    }

    /// Run forward kinematics, writing global coordinates for every atom.
    ///
    /// `conf.position` is the translation of the reference frame and
    /// `conf.orientation` a rotation vector; both are relative to the
    /// coordinates that were present when the tree was built.
    pub fn apply(&self, conf: &Conformation, out: &mut [Vec3]) {
        debug_assert_eq!(out.len(), self.reference.len());
        let frames = self.global_frames(conf);
        for i in 0..out.len() {
            out[i] = to_v3(&self.atom_position(&frames, i));
        }
    }

    /// Forward kinematics into a freshly allocated vector.
    pub fn coordinates(&self, conf: &Conformation) -> Vec<Vec3> {
        let mut out = vec![[0.0; 3]; self.reference.len()];
        self.apply(conf, &mut out);
        out
    }

    /// Atom indices belonging to the subtree rooted at `cluster`, including
    /// `cluster` itself. Used to accumulate torsion gradients.
    pub fn subtree_atoms(&self, cluster: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![cluster];
        while let Some(c) = stack.pop() {
            out.extend_from_slice(&self.clusters[c].atoms);
            stack.extend_from_slice(&self.clusters[c].children);
        }
        out.sort_unstable();
        out
    }

    /// Precomputed subtree atom lists for every cluster, for the hot loop.
    pub fn subtree_table(&self) -> Vec<Vec<usize>> {
        (0..self.clusters.len())
            .map(|c| self.subtree_atoms(c))
            .collect()
    }

    /// Global pivot point and unit axis direction of the torsion that joins
    /// cluster `ci` to its parent, or `None` when the cluster has no torsion
    /// (a rigidly welded component).
    ///
    /// Shared by [`Self::conf_gradient`] and the frame construction so the two
    /// can never disagree about where the rotation axis is.
    fn torsion_axis_in_world(
        &self,
        ci: usize,
        frames: &(Vec<Vector3<f64>>, Vec<Matrix3<f64>>),
    ) -> Option<(Vector3<f64>, Vector3<f64>)> {
        let (origin_global, rot_global) = frames;
        let cl = &self.clusters[ci];
        let parent = cl.parent?;
        let t_idx = cl.torsion?;
        let o_parent = v3(self.clusters[parent].origin);
        let to_world =
            |local: Vector3<f64>| origin_global[parent] + rot_global[parent] * (local - o_parent);
        let da = cl.axis_distal?;
        let pivot = to_world(v3(self.reference[da]));
        let axis = rot_global[parent] * v3(self.axis_dirs[t_idx]);
        Some((pivot, axis))
    }

    /// Map a gradient with respect to atom positions onto the degrees of
    /// freedom of `conf`.
    ///
    /// The returned vector has `6 + num_torsions` entries ordered
    /// `[∂/∂tx, ∂/∂ty, ∂/∂tz, ∂/∂θx, ∂/∂θy, ∂/∂θz, ∂/∂τ₀, …]`.
    ///
    /// # Rotational part
    ///
    /// With `R = exp([θ]×)` the gradient of a scalar field under a change of
    /// attitude is the **Riemannian (right-trivialised) gradient**
    ///
    /// ```text
    /// ∇_θ f = J_r(θ)ᵀ · (Rᵀ τ),    τ = Σ_i (R p_i) × g_i
    /// ```
    ///
    /// where `τ` is the spatial torque about the reference origin and `J_r` the
    /// right Jacobian of SO(3). This is exact at every rotation angle — unlike
    /// the common shortcut of using `Rᵀτ` alone, which is only correct in the
    /// limit of small `θ`. The sign convention is pinned by
    /// `conf_gradient_matches_finite_difference_everywhere`, which checks every
    /// degree of freedom against central differences.
    ///
    /// # Torsion part
    ///
    /// Each torsion is a single scalar degree of freedom, so its gradient is
    /// the torque about that bond's axis projected onto the axis direction.
    pub fn conf_gradient(&self, conf: &Conformation, atom_grad: &[Vector3<f64>]) -> Vec<f64> {
        let r_body = rotvec_to_matrix(&v3(conf.orientation));
        // Positions in the ligand-local frame, after the torsions but before
        // the rigid transform. These are exactly the lever arms of a rigid
        // rotation, which is why the body/torsion split matters.
        let (body_origin, body_rot) = self.body_frames(conf);
        let frames = self.global_frames(conf);

        // ---- Rigid-body translation ---------------------------------------
        // Every atom translates with the body, so the Jacobian is the identity.
        let mut grad_t = Vector3::zeros();
        for g in atom_grad.iter().take(self.reference.len()) {
            grad_t += g;
        }

        // ---- Rigid-body orientation ---------------------------------------
        // Spatial torque about the body-rotation centre: lever arm r_i − t.
        let mut torque = Vector3::zeros();
        for (i, atom) in self.atom_cluster.iter().enumerate() {
            let m = body_origin[*atom]
                + body_rot[*atom] * (v3(self.reference[i]) - v3(self.clusters[*atom].origin));
            torque += (r_body * m).cross(&atom_grad[i]);
        }
        let jr = so3_right_jacobian(&v3(conf.orientation));
        let grad_theta: Vector3<f64> = jr.transpose() * (r_body.transpose() * torque);

        let mut out = Vec::with_capacity(6 + self.torsions.len());
        out.extend_from_slice(&to_v3(&grad_t));
        out.extend_from_slice(&to_v3(&grad_theta));

        // ---- Torsions ------------------------------------------------------
        // dE/dτ = axis · Σ_{i ∈ subtree} (r_i − pivot) × g_i
        for (ci, cl) in self.clusters.iter().enumerate() {
            if cl.torsion.is_none() {
                continue;
            }
            let Some((pivot, axis)) = self.torsion_axis_in_world(ci, &frames) else {
                continue;
            };

            let mut sub_torque = Vector3::zeros();
            accumulate_torque(self, ci, &frames, &pivot, atom_grad, &mut sub_torque);
            out.push(axis.dot(&sub_torque));
        }
        out
    }

    /// Global position of a single atom under `conf` (convenience wrapper).
    pub fn to_global(&self, conf: &Conformation, atom: usize) -> Vector3<f64> {
        let frames = self.global_frames(conf);
        self.atom_position(&frames, atom)
    }

    /// Draw a random conformation.
    ///
    /// Torsions are uniform on the circle; the orientation is a uniformly
    /// distributed random rotation. The translation is left at the origin so
    /// that the caller places the ligand inside the search box.
    pub fn randomize(&self, rng: &mut impl Rng) -> Conformation {
        let torsions = (0..self.torsions.len())
            .map(|_| rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI))
            .collect();
        // Shoemake's uniform random quaternion.
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let u3: f64 = rng.gen();
        let s1 = (1.0 - u1).sqrt();
        let s2 = u1.sqrt();
        let q = Quaternion {
            w: s2 * (2.0 * std::f64::consts::PI * u3).cos(),
            x: s1 * (2.0 * std::f64::consts::PI * u2).cos(),
            y: s1 * (2.0 * std::f64::consts::PI * u2).sin(),
            z: s2 * (2.0 * std::f64::consts::PI * u3).sin(),
        };
        Conformation {
            position: [0.0, 0.0, 0.0],
            orientation: to_v3(&q.to_rotvec()),
            torsions,
        }
    }
}

/// Recursively accumulate the torque of the subtree rooted at `cluster`
/// about `axis_point`.
///
/// Positions come from [`KinematicTree::atom_position`] so that the per-cluster
/// local-origin offset is applied *inside* the cluster's rotation; subtracting
/// it in world coordinates afterwards would be wrong for every non-root
/// cluster.
fn accumulate_torque(
    tree: &KinematicTree,
    cluster: usize,
    frames: &(Vec<Vector3<f64>>, Vec<Matrix3<f64>>),
    axis_point: &Vector3<f64>,
    atom_grad: &[Vector3<f64>],
    acc: &mut Vector3<f64>,
) {
    for &atom in &tree.clusters[cluster].atoms {
        let r = tree.atom_position(frames, atom);
        *acc += (r - axis_point).cross(&atom_grad[atom]);
    }
    for &child in &tree.clusters[cluster].children {
        accumulate_torque(tree, child, frames, axis_point, atom_grad, acc);
    }
}

/// Determine which bonds are rotatable.
///
/// A bond is rotatable when all of the following hold:
///
/// * it is **not** in a ring — breaking a ring bond would require distorting
///   a covalent bond;
/// * both atoms are heavy (hydrogen bonds are never torsion axes);
/// * neither atom is **terminal** — each keeps at least one other heavy
///   neighbour, so the rotation actually moves a fragment;
/// * it is **not an amide** (or the sulfur/thioamide analogue) — the partial
///   double-bond character of `C(=O)–N` makes the torsion effectively rigid,
///   and AutoDock treats it as such.
///
/// Terminal methyls are excluded by the "non-terminal" rule, matching the
/// AutoDock convention that they contribute no rotatable degree of freedom.
pub fn perceive_rotatable_bonds(mol: &Molecule) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for bond in &mol.bonds {
        if bond.in_ring {
            continue;
        }
        let (i, j) = (bond.i, bond.j);
        if mol.atoms[i].element == Element::H || mol.atoms[j].element == Element::H {
            continue;
        }
        if mol.atoms[i].is_metal() || mol.atoms[j].is_metal() {
            continue;
        }
        if heavy_degree(mol, i) < 2 || heavy_degree(mol, j) < 2 {
            continue;
        }
        if is_amide_like(mol, i, j) {
            continue;
        }
        out.push((i, j));
    }
    // A bond listed twice in `mol.bonds` is one bond, and returning it twice
    // would double every torsion downstream.
    out.sort_unstable();
    out.dedup();
    out
}

fn heavy_degree(mol: &Molecule, i: usize) -> usize {
    mol.neighbors[i]
        .iter()
        .filter(|&&j| mol.atoms[j].element != Element::H)
        .count()
}

/// True if the `(i, j)` bond is an amide / thioamide C–N or C–S bond.
///
/// Detection is geometric and type-based: a nitrogen bonded to a carbon that
/// also bears a short bond to an oxygen or sulfur acceptor is in an amide.
fn is_amide_like(mol: &Molecule, i: usize, j: usize) -> bool {
    let (n, c) = if mol.atoms[i].element == Element::N {
        (i, j)
    } else if mol.atoms[j].element == Element::N {
        (j, i)
    } else {
        return false;
    };
    if mol.atoms[c].element != Element::C {
        return false;
    }
    for &o in &mol.neighbors[c] {
        if o == n {
            continue;
        }
        let el = mol.atoms[o].element;
        if (el == Element::O || el == Element::S)
            && mol.atoms[o].atom_type.is_acceptor()
            && dist3(mol.atoms[c].coord, mol.atoms[o].coord) < 1.45
        {
            return true;
        }
    }
    false
}

/// A minimal union-find over atom indices.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> UnionFind {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Atom, AtomType};

    /// n-butane: four carbons in a zig-zag, two rotatable C–C bonds.
    fn butane() -> Molecule {
        let c = |i: u32, p: Vec3| Atom::new(i, p, Element::C, AtomType::CH);
        Molecule::from_atoms(vec![
            c(1, [0.000, 0.000, 0.000]),
            c(2, [1.500, 0.000, 0.000]),
            c(3, [2.150, 1.300, 0.000]),
            c(4, [3.650, 1.300, 0.000]),
        ])
        .unwrap()
    }

    #[test]
    fn perceives_one_rotatable_bond_in_butane() {
        // n-Butane has exactly one rotatable bond (C2–C3): the C1–C2 and C3–C4
        // bonds each terminate in a methyl, and AutoDock does not count a
        // terminal methyl as a rotatable degree of freedom.
        let mol = butane();
        let rot = perceive_rotatable_bonds(&mol);
        assert_eq!(rot, vec![(1, 2)], "got {rot:?}");
    }

    #[test]
    fn ethane_has_no_rotatable_bonds() {
        let mol = Molecule::from_atoms(vec![
            Atom::new(1, [0.0, 0.0, 0.0], Element::C, AtomType::CH),
            Atom::new(2, [1.5, 0.0, 0.0], Element::C, AtomType::CH),
        ])
        .unwrap();
        assert!(perceive_rotatable_bonds(&mol).is_empty());
    }

    /// Ibuprofen's heavy-atom graph, exactly as RDKit hands it to the engine.
    fn ibuprofen_graph() -> Molecule {
        let coords: Vec<Vec3> = vec![
            [4.650, -0.310, 0.616],
            [3.182, 0.118, 0.560],
            [3.071, 1.492, -0.106],
            [2.354, -0.951, -0.180],
            [0.862, -0.729, -0.104],
            [0.170, -0.953, 1.092],
            [-1.210, -0.742, 1.165],
            [-1.927, -0.304, 0.044],
            [-1.230, -0.079, -1.152],
            [0.149, -0.290, -1.225],
            [-3.425, -0.078, 0.141],
            [-4.205, -1.012, -0.787],
            [-3.796, 1.367, -0.133],
            [-4.103, 1.858, -1.208],
            [-3.744, 2.139, 0.974],
        ];
        let types = [
            AtomType::CH,
            AtomType::CH,
            AtomType::CH,
            AtomType::CH,
            AtomType::CP,
            AtomType::CP,
            AtomType::CP,
            AtomType::CP,
            AtomType::CP,
            AtomType::CP,
            AtomType::CH,
            AtomType::CH,
            AtomType::CP,
            AtomType::OA,
            AtomType::OA,
        ];
        let atoms: Vec<Atom> = coords
            .iter()
            .zip(types)
            .enumerate()
            .map(|(i, (p, t))| {
                let element = if matches!(t, AtomType::OA) {
                    Element::O
                } else {
                    Element::C
                };
                Atom::new(i as u32 + 1, *p, element, t)
            })
            .collect();
        let bonds: Vec<(usize, usize)> = vec![
            (0, 1),
            (1, 2),
            (1, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (7, 10),
            (10, 11),
            (10, 12),
            (12, 13),
            (12, 14),
            (4, 9),
        ];
        let mut mol = Molecule {
            atoms,
            bonds: Vec::new(),
            neighbors: Vec::new(),
        };
        for (i, j) in bonds {
            mol.bonds.push(crate::types::Bond::new(i, j));
        }
        let n = mol.atoms.len();
        mol.neighbors = vec![Vec::new(); n];
        for bond in &mol.bonds {
            if !mol.neighbors[bond.i].contains(&bond.j) {
                mol.neighbors[bond.i].push(bond.j);
                mol.neighbors[bond.j].push(bond.i);
            }
        }
        mol.assign_ring_membership().unwrap();
        mol
    }

    #[test]
    fn ibuprofen_has_four_rotatable_bonds() {
        let mol = ibuprofen_graph();
        assert_eq!(perceive_rotatable_bonds(&mol).len(), 4);
    }

    #[test]
    fn duplicate_bonds_do_not_duplicate_torsions() {
        // A bond listed twice must not become two torsions.
        //
        // This is not hypothetical: the PyO3 front-end once appended the
        // caller's RDKit bond table to the distance-perceived one, and the
        // duplicated rotatable bonds made the cluster graph cyclic, so the BFS
        // could not give every torsion its own child cluster. That hit an
        // `expect` inside a `panic = "abort"` binary and killed the caller's
        // Python process outright.
        let mut mol = ibuprofen_graph();
        let doubled: Vec<crate::types::Bond> = mol.bonds.clone();
        mol.bonds.extend(doubled);
        // `neighbors` is deduplicated, as the binding layer now does.
        mol.assign_ring_membership().unwrap();

        let rot = perceive_rotatable_bonds(&mol);
        assert_eq!(
            rot.len(),
            4,
            "duplicated bonds leaked into the torsion list: {rot:?}"
        );

        let tree = KinematicTree::from_molecule(&mol, &[]).expect("tree");
        assert_eq!(tree.num_torsions(), 4);
        assert_eq!(tree.clusters.len(), 5);
    }

    #[test]
    fn disconnected_molecule_does_not_panic() {
        // Two fragments, far apart: the second is attached rigidly.
        let mut mol = butane();
        let extra = Atom::new(5, [50.0, 0.0, 0.0], Element::O, AtomType::OA);
        mol.atoms.push(extra);
        let n = mol.atoms.len();
        mol.bonds.clear();
        mol.neighbors = vec![Vec::new(); n];
        for bond in [(0usize, 1usize), (1, 2), (2, 3)] {
            mol.bonds.push(crate::types::Bond::new(bond.0, bond.1));
            mol.neighbors[bond.0].push(bond.1);
            mol.neighbors[bond.1].push(bond.0);
        }
        mol.assign_ring_membership().unwrap();

        let tree = KinematicTree::from_molecule(&mol, &[]).expect("tree builds");
        assert_eq!(tree.num_torsions(), 1, "the butane torsion survives");
        assert_eq!(
            tree.clusters.len(),
            3,
            "butane clusters plus the isolated atom"
        );
        // The stray atom gets its own rigid cluster pinned to the root, with no
        // torsion of its own — there is no bond to rotate about.
        let stray = tree.atom_cluster[4];
        assert_ne!(stray, tree.root);
        assert_eq!(tree.clusters[stray].parent, Some(tree.root));
        assert_eq!(tree.clusters[stray].torsion, None);
        // It still moves with the rigid body.
        let conf = Conformation {
            position: [1.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0],
            torsions: vec![0.0],
        };
        let coords = tree.coordinates(&conf);
        assert!((coords[4][0] - 51.0).abs() < 1e-9, "got {coords:?}");
    }

    #[test]
    fn a_cyclic_torsion_list_is_repaired_not_fatal() {
        // Declaring a *ring* bond as a torsion must not blow up.
        //
        // A preparer that lists a ring bond as rotatable is wrong, but the
        // engine can still do something sensible: the bond is contracted into
        // the rigid cluster, exactly as it would have been had it been
        // perceived from the graph.
        let mol = ibuprofen_graph();
        let declared = vec![
            (0usize, 1usize),
            (1usize, 3usize),
            (3, 4),
            (7, 10),
            (10, 12),
            (4, 9),
        ];
        let tree = KinematicTree::from_molecule(&mol, &declared).expect("tree builds");
        // (4, 9) closes the benzene ring, so it is contracted away.
        assert_eq!(tree.num_torsions(), 5);
        assert_eq!(tree.clusters.len(), 6);
    }

    #[test]
    fn benzene_has_no_rotatable_bonds() {
        // Regular hexagon, 1.39 Å edges.
        let atoms = (0..6)
            .map(|i| {
                let a = i as f64 * std::f64::consts::PI / 3.0;
                Atom::new(
                    i + 1,
                    [1.39 * a.cos(), 1.39 * a.sin(), 0.0],
                    Element::C,
                    AtomType::CP,
                )
            })
            .collect();
        let mol = Molecule::from_atoms(atoms).unwrap();
        assert!(perceive_rotatable_bonds(&mol).is_empty());
        assert!(mol.bonds.iter().all(|b| b.in_ring));
    }

    #[test]
    fn amide_bond_is_not_rotatable() {
        // O=C–N: the C–N bond must be excluded.
        let a = |i: u32, p: Vec3, e: Element, t: AtomType| Atom::new(i, p, e, t);
        let mol = Molecule::from_atoms(vec![
            a(1, [0.000, 0.000, 0.000], Element::C, AtomType::CH),
            a(2, [1.230, 0.600, 0.000], Element::O, AtomType::OA),
            a(3, [1.230, -0.750, 0.000], Element::N, AtomType::NA),
            a(4, [2.400, -1.400, 0.000], Element::C, AtomType::CH),
        ])
        .unwrap();
        let rot = perceive_rotatable_bonds(&mol);
        assert!(
            !rot.contains(&(2, 3)),
            "amide C–N must not be rotatable, got {rot:?}"
        );
    }

    #[test]
    fn tree_covers_every_atom() {
        let mol = butane();
        let tree = KinematicTree::from_molecule(&mol, &[]).unwrap();
        assert_eq!(tree.num_torsions(), 1);
        assert_eq!(tree.clusters.len(), 2);
        let total: usize = tree.clusters.iter().map(|c| c.atoms.len()).sum();
        assert_eq!(total, mol.len());
        // The root is the larger fragment: C1 and C2.
        assert_eq!(tree.clusters[0].atoms, vec![0, 1]);
        assert_eq!(tree.clusters[1].atoms, vec![2, 3]);
    }

    #[test]
    fn identity_conformation_reproduces_reference() {
        let mol = butane();
        let tree = KinematicTree::from_molecule(&mol, &[]).unwrap();
        let conf = Conformation::identity(tree.num_torsions());
        let out = tree.coordinates(&conf);
        for (a, b) in mol.atoms.iter().zip(out.iter()) {
            for k in 0..3 {
                assert!(
                    (a.coord[k] - b[k]).abs() < 1e-9,
                    "atom {} axis {k}: {} vs {}",
                    a.serial,
                    a.coord[k],
                    b[k]
                );
            }
        }
    }

    #[test]
    fn rigid_body_translation_moves_every_atom() {
        let mol = butane();
        let tree = KinematicTree::from_molecule(&mol, &[]).unwrap();
        let mut conf = Conformation::identity(tree.num_torsions());
        conf.position = [1.0, 2.0, 3.0];
        let out = tree.coordinates(&conf);
        for (a, b) in mol.atoms.iter().zip(out.iter()) {
            for k in 0..3 {
                assert!((a.coord[k] + conf.position[k] - b[k]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn rotating_torsion_keeps_bond_length_and_moves_distal_atoms() {
        let mol = butane();
        let tree = KinematicTree::from_molecule(&mol, &[]).unwrap();
        assert_eq!(tree.num_torsions(), 1);
        let base = tree.coordinates(&Conformation::identity(tree.num_torsions()));
        let mut conf = Conformation::identity(tree.num_torsions());
        conf.torsions[0] = std::f64::consts::FRAC_PI_2;
        let rotated = tree.coordinates(&conf);

        // The C2–C3 bond length must be invariant.
        let l0 = dist3(base[1], base[2]);
        let l1 = dist3(rotated[1], rotated[2]);
        assert!((l0 - l1).abs() < 1e-9, "bond length changed {l0} -> {l1}");
        assert!(l0 > 1.4 && l0 < 1.5, "unexpected bond length {l0}");

        // Proximal atoms (cluster 0) stay put; distal atoms must move.
        for k in 0..3 {
            assert!((base[0][k] - rotated[0][k]).abs() < 1e-12);
            assert!((base[1][k] - rotated[1][k]).abs() < 1e-12);
        }
        assert!(dist3(base[3], rotated[3]) > 0.5, "distal atom should move");
    }

    #[test]
    fn rigid_rotation_preserves_internal_distances() {
        let mol = butane();
        let tree = KinematicTree::from_molecule(&mol, &[]).unwrap();
        let mut conf = Conformation::identity(tree.num_torsions());
        conf.orientation = [0.3, -0.5, 0.9];
        conf.position = [1.0, 2.0, 3.0];
        let out = tree.coordinates(&conf);
        // Every pairwise distance must be preserved by a rigid transform.
        for i in 0..mol.len() {
            for j in (i + 1)..mol.len() {
                let d0 = dist3(mol.atoms[i].coord, mol.atoms[j].coord);
                let d1 = dist3(out[i], out[j]);
                assert!((d0 - d1).abs() < 1e-9, "pair {i}-{j}: {d0} vs {d1}");
            }
        }
    }

    #[test]
    fn rotation_vector_to_matrix_is_orthonormal() {
        for theta in [
            Vector3::new(0.001, 0.0, 0.0),
            Vector3::new(0.3, 0.4, 0.5),
            Vector3::new(3.0, 0.0, 0.0),
        ] {
            let r = rotvec_to_matrix(&theta);
            let should_be_identity = r.transpose() * r;
            for i in 0..3 {
                for j in 0..3 {
                    let e = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (should_be_identity[(i, j)] - e).abs() < 1e-12,
                        "R not orthonormal at {theta:?}: {}",
                        should_be_identity[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn quaternion_round_trips_through_rotvec() {
        // Only angles with |θ| < π round-trip: to_rotvec canonicalises to the
        // shorter arc, which is the correct behaviour for a rotation vector.
        for theta in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.3, 0.4, 0.5),
            Vector3::new(-1.0, 0.6, 0.2),
            Vector3::new(2.5, 0.0, 0.0),
        ] {
            let q = Quaternion::from_rotvec(&theta);
            let back = q.to_rotvec();
            assert!((back - theta).norm() < 1e-9, "{back:?} vs {theta:?}");
        }
        // Beyond π the quaternion must still describe the same rotation.
        let theta = Vector3::new(4.0, 0.0, 0.0);
        let q = Quaternion::from_rotvec(&theta);
        let expected = rotvec_to_matrix(&theta);
        for i in 0..3 {
            for j in 0..3 {
                assert!((q.to_matrix()[(i, j)] - expected[(i, j)]).abs() < 1e-12);
            }
        }
    }

    /// The translational part of `conf_gradient` must reproduce the plain
    /// sum of the atom gradients.
    #[test]
    fn conf_gradient_translation_matches_direct_sum() {
        let mol = butane();
        let tree = KinematicTree::from_molecule(&mol, &[]).unwrap();
        let conf = Conformation::identity(tree.num_torsions());
        let grads: Vec<Vector3<f64>> = (0..mol.len())
            .map(|i| {
                let f = i as f64;
                Vector3::new(f, 2.0 * f, -f)
            })
            .collect();
        let g = tree.conf_gradient(&conf, &grads);
        let mut expect = Vector3::zeros();
        for gr in &grads {
            expect += gr;
        }
        assert!((Vector3::new(g[0], g[1], g[2]) - expect).norm() < 1e-12);
    }

    /// A rigid-body rotation gradient must be zero when the artificial force
    /// field is itself rotationally symmetric about the reference origin.
    #[test]
    fn conf_gradient_rotation_is_zero_for_central_field() {
        let mol = butane();
        let tree = KinematicTree::from_molecule(&mol, &[]).unwrap();
        let conf = Conformation::identity(tree.num_torsions());
        // Radial gradients produce no torque about the origin.
        let grads: Vec<Vector3<f64>> = mol.atoms.iter().map(|a| v3(a.coord) * -0.5).collect();
        let g = tree.conf_gradient(&conf, &grads);
        let rot = Vector3::new(g[3], g[4], g[5]);
        assert!(rot.norm() < 1e-12, "expected zero torque, got {rot:?}");
    }

    /// Compare the **entire** analytic `conf_gradient` against central finite
    /// differences, at several conformations. This is the strongest available
    /// check on the rigid-body Jacobian, the SO(3) pull-back and the torsion
    /// torques all at once.
    #[test]
    fn conf_gradient_matches_finite_difference_everywhere() {
        let mol = butane();
        let tree = KinematicTree::from_molecule(&mol, &[]).unwrap();
        let n = tree.num_torsions();

        // A deliberately awkward, non-separable, non-central force field.
        let grads: Vec<Vector3<f64>> = (0..mol.len())
            .map(|i| {
                let f = i as f64;
                Vector3::new(
                    0.3 * (f + 1.0) - 0.11 * f * f,
                    -0.7 + 0.2 * f,
                    1.1 - 0.2 * f + 0.05 * f * f * f,
                )
            })
            .collect();
        // E(conf) = Σ_i g_i · r_i(conf) — a linear functional of the
        // coordinates, so its gradient w.r.t. any DOF is exactly what
        // `conf_gradient` computes.
        let energy = |c: &Conformation| -> f64 {
            tree.coordinates(c)
                .iter()
                .zip(grads.iter())
                .map(|(r, g)| g.x * r[0] + g.y * r[1] + g.z * r[2])
                .sum()
        };

        let cases = [
            Conformation {
                position: [0.0, 0.0, 0.0],
                orientation: [0.0, 0.0, 0.0],
                torsions: vec![0.0; n],
            },
            Conformation {
                position: [1.5, -2.0, 0.5],
                orientation: [0.0, 0.0, 0.0],
                torsions: vec![0.0; n],
            },
            // Large angles exercise the right-Jacobian pull-back, which the
            // naive -Rᵀτ shortcut gets wrong.
            Conformation {
                position: [0.0, 0.0, 0.0],
                orientation: [1.9, 0.4, -0.7],
                torsions: vec![0.0; n],
            },
            Conformation {
                position: [0.0, 0.0, 0.0],
                orientation: [0.3, 0.4, 0.5],
                torsions: vec![0.4; n],
            },
        ];

        let h = 1e-6;
        for (ci, conf) in cases.iter().enumerate() {
            let g = tree.conf_gradient(conf, &grads);
            assert_eq!(g.len(), 6 + n, "gradient length must be 6 + #torsions");

            let mut numeric = vec![0.0; 6 + n];
            for k in 0..3 {
                let mut cp = conf.clone();
                cp.position[k] += h;
                let mut cm = conf.clone();
                cm.position[k] -= h;
                numeric[k] = (energy(&cp) - energy(&cm)) / (2.0 * h);
            }
            for k in 0..3 {
                let mut cp = conf.clone();
                cp.orientation[k] += h;
                let mut cm = conf.clone();
                cm.orientation[k] -= h;
                numeric[3 + k] = (energy(&cp) - energy(&cm)) / (2.0 * h);
            }
            for t in 0..n {
                let mut cp = conf.clone();
                cp.torsions[t] += h;
                let mut cm = conf.clone();
                cm.torsions[t] -= h;
                numeric[6 + t] = (energy(&cp) - energy(&cm)) / (2.0 * h);
            }

            for k in 0..6 + n {
                assert!(
                    (numeric[k] - g[k]).abs() < 1e-4,
                    "case {ci}, dof {k}: numeric {:.8} vs analytic {:.8}",
                    numeric[k],
                    g[k]
                );
            }
        }
    }

    #[test]
    fn declared_torsions_are_honoured() {
        let mol = butane();
        let tree = KinematicTree::from_molecule(&mol, &[(0, 1)]).unwrap();
        assert_eq!(tree.num_torsions(), 1);
    }
}
