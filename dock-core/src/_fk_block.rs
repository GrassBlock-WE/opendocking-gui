    /// Compute the global frame (origin + rotation) of every rigid cluster.
    ///
    /// This is the single source of truth for forward kinematics: [`Self::apply`],
    /// [`Self::conf_gradient`] and [`Self::to_global`] all route through it, so
    /// they cannot drift apart.
    pub fn global_frames(&self, conf: &Conformation) -> (Vec<Vector3<f64>>, Vec<Matrix3<f64>>) {
        let n = self.clusters.len();
        let mut origin_global: Vec<Vector3<f64>> = vec![Vector3::zeros(); n];
        let mut rot_global: Vec<Matrix3<f64>> = vec![Matrix3::identity(); n];

        // Root: the rigid-body transform, relative to the reference frame.
        let r_body = rotvec_to_matrix(&v3(conf.orientation));
        let t_body = v3(conf.position);
        origin_global[0] = r_body * v3(self.clusters[0].origin) + t_body;
        rot_global[0] = r_body;

        // Descendants, in topological order (a parent always precedes a child).
        for ci in 1..n {
            let cl = &self.clusters[ci];
            let parent = cl.parent.expect("non-root cluster has a parent");
            let ti = cl.torsion.expect("non-root cluster has a torsion");
            let (pa, da) = (
                cl.axis_proximal.expect("axis"),
                cl.axis_distal.expect("axis"),
            );
            let ref_pa = v3(self.reference[pa]);
            let ref_da = v3(self.reference[da]);
            let pivot = origin_global[parent] + rot_global[parent] * ref_da;
            let direction = (pivot - (origin_global[parent] + rot_global[parent] * ref_pa)).normalize();
            let angle = conf.torsions.get(ti).copied().unwrap_or(0.0);
            let local_rot = rotvec_to_matrix(&(direction * angle));
            rot_global[ci] = rot_global[parent] * local_rot;
            origin_global[ci] = pivot + local_rot * (origin_global[ci] - pivot);
        }
        (origin_global, rot_global)
    }

    /// Global position of one atom under `conf`.
    pub fn atom_position(&self, frames: &(Vec<Vector3<f64>>, Vec<Matrix3<f64>>), atom: usize) -> Vector3<f64> {
        let (origin_global, rot_global) = frames;
        let ci = self.atom_cluster[atom];
        origin_global[ci] + rot_global[ci] * (v3(self.reference[atom]) - v3(self.clusters[ci].origin))
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

    /// Map a gradient with respect to atom positions onto the degrees of
    /// freedom of `conf`.
    ///
    /// The returned vector is ordered `[∂/∂tx, ∂/∂ty, ∂/∂tz, ∂/∂θx, ∂/∂θy, ∂/∂θz,
    /// ∂/∂θ_torsion 0, …]` with one entry per three degrees of freedom.
    ///
    /// The rotational part is the **Riemannian gradient** on SO(3): the spatial
    /// torque is rotated into the body frame and then pulled back through the
    /// transpose of the right Jacobian, so it stays exact at any rotation angle
    /// rather than only for small perturbations.
    pub fn conf_gradient(
        &self,
        conf: &Conformation,
        atom_grad: &[Vector3<f64>],
    ) -> Vec<f64> {
        // Translation: every atom moves with the rigid body, so the Jacobian is I.
        let mut grad_t = Vector3::zeros();
        // G_ab = Σ_i g_{i,a} · p_{i,b}  (body-frame position p = reference coord).
        let mut g_mat = Matrix3::zeros();
        for (i, p) in self.reference.iter().enumerate() {
            let gi = atom_grad[i];
            for a in 0..3 {
                grad_t[a] += gi[a];
                for b in 0..3 {
                    g_mat[(a, b)] += gi[a] * p[b];
                }
            }
        }
        // ∇_θ = G · Rᵀ · J_r(θ)ᵀ
        let r_body = rotvec_to_matrix(&v3(conf.orientation));
        let jr = so3_right_jacobian(&v3(conf.orientation));
        let grad_theta = g_mat * r_body.transpose() * jr.transpose();

        let mut out = Vec::with_capacity(6 + 3 * self.torsions.len());
        out.extend_from_slice(&grad_t);
        out.extend_from_slice(&grad_theta);

        // Torsions: torque about the bond axis, summed over the distal subtree.
        // The frames are computed once, keeping this linear in the torsion count.
        let frames = self.global_frames(conf);
        let (origin_global, rot_global) = &frames;
        for cl in self.clusters.iter().skip(1) {
            let parent = cl.parent.expect("non-root");
            let (pa, da) = (
                cl.axis_proximal.expect("axis"),
                cl.axis_distal.expect("axis"),
            );
            let axis_point = origin_global[parent] + rot_global[parent] * v3(self.reference[da]);
            let axis_world =
                (rot_global[parent] * (v3(self.reference[da]) - v3(self.reference[pa]))).normalize();

            let mut torque = Vector3::zeros();
            // Walk the subtree through the child links.
            let mut stack = vec![self.atom_cluster[da]];
            let mut seen = std::collections::HashSet::new();
            stack.reverse(); // process in ascending order for determinism
            while let Some(c) = stack.pop() {
                if !seen.insert(c) {
                    continue;
                }
                for &atom in &self.clusters[c].atoms {
                    let r = self.atom_position(&frames, atom);
                    torque += (r - axis_point).cross(&atom_grad[atom]);
                }
                for &child in &self.clusters[c].children {
                    stack.push(child);
                }
            }
            out.extend_from_slice(&torque);
        }
        out
    }

    /// Global position of a single atom under `conf` (convenience wrapper).
    pub fn to_global(&self, conf: &Conformation, atom: usize) -> Vector3<f64> {
        let frames = self.global_frames(conf);
        self.atom_position(&frames, atom)
    }

