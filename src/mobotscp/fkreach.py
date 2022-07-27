#!/usr/bin/env python
import os
import h5py
import time
import baldor
import numpy as np
import openravepy as orpy
import tf.transformations as tr

##############################################################################################################
# Input: 
#   * robot model
#   * a set of orientations [sin(theta), 0, cos(theta)] with theta = polar angle
# Output: 
#   * raw Focus Kinematic Reachability (FKR): reachable points in discrete space relative to robot
#   * a reachable region with geometric shape based on raw FKR data
##############################################################################################################


########## Draw reachability limits in OpenRAVE ##########

class DrawReachLimits(object):
    def __init__(self, reach_param):
        self.spheres_center_wrt_base = reach_param.spheres_center_wrt_base
        self.radii = reach_param.radii
        self.Xmin = reach_param.Xmin
        self.Zmin = reach_param.Zmin
        self.Zmax = reach_param.Zmax

    def z_rotation(xyz, yaw_rad):
        rot_matrix = [  [np.cos(yaw_rad), -np.sin(yaw_rad), 0.],
                        [np.sin(yaw_rad), np.cos(yaw_rad),  0.],
                        [0.,              0.,               1.]  ]
        rotated_xyz = np.matmul(rot_matrix, np.transpose(xyz))
        return np.transpose(rotated_xyz)

    def draw_planes(self, env, base_pose):
        """
        Draw 2 planes into the OpenRAVE environment.
        Parameters
        ----------
        env: orpy.Environment
          The OpenRAVE environment
        base_pose: numpy.array
          Pose of the base: (x, y, yaw)
        Returns
        -------
        planes: orpy.GraphHandle
          Handles holding the plot.
        """
        base_pose = list(base_pose)
        # create 2 thin boxes to represent 2 limit planes
        if self.Zmax is not None:
            centers = [ [-0.001, 0, (self.Zmax-self.Zmin)/2.], [0.5, 0, -0.001], [0.5, 0, self.Zmax-self.Zmin+0.001] ]
            extents = [ [0.001, 1, (self.Zmax-self.Zmin)/2.],  [0.5, 1, 0.001],  [0.5, 1, 0.001] ]
        else:
            centers = [ [-0.001, 0, 0.75], [0.5, 0, -0.001] ]
            extents = [ [0.001, 1, 0.75],  [0.5, 1, 0.001] ]
        planes_center_wrt_base = np.array([self.Xmin, 0., self.Zmin])
        xyzabc = np.hstack((centers, extents))
        planes = orpy.RaveCreateKinBody(env, '')
        planes.SetName('planes')
        planes.InitFromBoxes(xyzabc, True)
        # move the boxes according to the robot's pose
        T = tr.euler_matrix(0, 0, base_pose[2], 'sxyz')
        T[:3, 3] = np.array(base_pose[:2]+[0]) + self.z_rotation(planes_center_wrt_base, base_pose[2])
        planes.SetTransform(T)
        for geom in planes.GetLinks()[0].GetGeometries():
            geom.SetTransparency(0.8)
            geom.SetDiffuseColor(color=((0.2,0.2,0.2)))
        with env:
            env.AddKinBody(planes)
        return planes

    def draw_spheres(self, env, base_pose):
        """
        Draw N spheres into the OpenRAVE environment.
        Parameters
        ----------
        env: orpy.Environment
          The OpenRAVE environment
        spheres_center_wrt_base: numpy.array 3 * N
          Centers of the spheres w.r.t. the base: [[x1, y1, z1], ..., [xN, yN, zN]]
        radii: numpy.array 1 * N 
          Radii of the spheres: [[r1], ..., [rN]]
        base_pose: numpy.array
          Pose of the base: (x, y, yaw)
        Returns
        -------
        spheres: orpy.GraphHandle
          Handles holding the plot.
        """
        base_pose = list(base_pose)
        sphere_i_center = np.array(base_pose[:2]+[0]) + self.z_rotation(self.spheres_center_wrt_base, base_pose[2])
        centers = []
        for i in range(len(self.radii)):
            centers.append(sphere_i_center)
        xyzr = np.hstack((centers, self.radii))
        spheres = orpy.RaveCreateKinBody(env, '')
        spheres.SetName('spheres')
        spheres.InitFromSpheres(xyzr, True)
        for geom in spheres.GetLinks()[0].GetGeometries():
            geom.SetTransparency(0.85)
            geom.SetDiffuseColor(color=((0.,1.,0.)))
        with env:
            env.AddKinBody(spheres)
        return spheres

    def remove_limits(self, env, obj1=None, obj2=None):
        with env:
            if obj1 is not None:
                env.Remove(obj1)
            if obj2 is not None:
                env.Remove(obj2)

    def draw_limits(self, env, base_pose):
        """
        Use both draw_planes and draw_spheres.
        """
        if env.GetKinBody('planes') is not None:
            self.remove_limits(env, env.GetKinBody('planes'))
        if env.GetKinBody('spheres') is not None:
            self.remove_limits(env, env.GetKinBody('spheres'))
        planes = self.draw_planes(env, base_pose)
        spheres = self.draw_spheres(env, base_pose)
        return planes, spheres



########## Generate Focus Kinematic Reachability ##########
########## Solve for reachability limits         ##########
########## Visualize FKR and limits in mayavi    ##########

def meter_from_voxel_unit(invoxel, fkr_xyzdelta):
    """Convert from reachability3d voxel unit to meter.
    """
    measure = invoxel * fkr_xyzdelta
    return measure

def voxel_unit_from_meter(measure, fkr_xyzdelta, unit_type=np.float64):
    """Convert from meter to reachability3d voxel unit.
    """
    invoxel = measure/fkr_xyzdelta
    return invoxel.astype(unit_type)

def get_link_offset(robot, l0_name='link0', l1_name='link1'):
    for link in robot.GetLinks():
        if l0_name in link.GetName():
            link_0 = link
        elif l1_name in link.GetName():
            link_1 = link
    p_l1 = link_1.GetTransform()[:3, 3]
    p_l0 = link_0.GetTransform()[:3, 3]
    return p_l1 - p_l0

def display_mayavi_ori_axes(xcolor=(1, 0, 0), ycolor=(0, 1, 0), zcolor=(0, 0, 1), line_width=3, scale_factor=10):
    from mayavi import mlab
    origin_x = mlab.quiver3d(0.2115, 0, 0.320, 1, 0, 0, color=xcolor, line_width=line_width,
                             scale_factor=scale_factor)
    origin_y = mlab.quiver3d(0.2115, 0, 0.320, 0, 1, 0, color=ycolor, line_width=line_width,
                             scale_factor=scale_factor)
    origin_z = mlab.quiver3d(0.2115, 0, 0.320, 0, 0, 1, color=zcolor, line_width=line_width,
                             scale_factor=scale_factor)
    return origin_x, origin_y, origin_z

def display_mayavi_viewpoint(view):
    print("Azimuth: {} | Elevation: {} | Distance: {} | Focalpoints: {}".format(view[0], view[1], view[2], view[3]))


class GenerateFKR(object):
    def __init__(self, base_approach_dir=[1., 0], sampling_mode="visible-front", xyz_delta=0.04, \
                 angle_inc=np.pi/6., angle_offset=0., max_radius=None, orientation_list=[], l1_from_ground=0.5175):
        self.base_approach_dir = base_approach_dir
        self.sampling_mode = sampling_mode
        self.xyzdelta = xyz_delta
        self.angle_inc = angle_inc
        self.angle_offset = angle_offset
        self.max_radius = max_radius
        self.sampling_dirs = orientation_list
        self.l1_from_ground = l1_from_ground


class FKRParameters(object):
    def __init__(self, data_id, gen_fkr_param=None):
        self.data_id = data_id
        self.gen_fkr_param = gen_fkr_param


class ReachLimitParameters(object):
    def __init__(self, Rmin=0.5, Rmax=1.0, Xmin_wrt_arm=0.5, Zmin_wrt_arm=0.75, Zmax_wrt_arm=None, \
                 spheres_center_wrt_arm=[0., 0., 0.1975], arm_ori_wrt_base=[0.2115, 0., 0.320]):
        # spheres' parameters
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.radii = [[Rmin], [Rmax]]
        self.spheres_center_wrt_arm = spheres_center_wrt_arm
        self.spheres_center_wrt_base = np.array(spheres_center_wrt_arm) + np.array(arm_ori_wrt_base)
        # planes' parameters
        self.Xmin_wrt_arm = Xmin_wrt_arm
        self.Xmin = Xmin_wrt_arm + arm_ori_wrt_base[0]
        self.Zmin_wrt_arm = Zmin_wrt_arm
        self.Zmin = Zmin_wrt_arm + arm_ori_wrt_base[2]
        self.Zmax_wrt_arm = Zmax_wrt_arm
        if Zmax_wrt_arm is not None:
            self.Zmax = Zmax_wrt_arm + arm_ori_wrt_base[2]
        else:
            self.Zmax = None


class FocusKinematicReachability(orpy.databases.kinematicreachability.ReachabilityModel):
    def __init__(self, env, robot, param):
        self.env = env
        self.robot = robot
        try:
            self.manip = self.robot.GetActiveManipulator()
        except:
            raise ValueError("Active manipulator is not set")
        self.data_id = param.data_id
        # FKR
        self.fkr_version = "1.3.0"
        self.fkr_3d = None
        self.fkr_qdensity3d = None
        self.fkr_rotdensity3d = None
        self.fkr_extent = None
        self._fkr_databasefile = None
        self.fkr_build_time = None

        super(FocusKinematicReachability, self).__init__(robot=self.robot)
        self.load()

        # Check FKR dataset availability
        if param.gen_fkr_param is not None:
            if self.has_file_name(self.data_id):
                print("Database id [{}] is already available. Overwritting is not currently supported.")
            else:
                gen_ = param.gen_fkr_param
                print("Database id [{}] is not found. Generating with params:".format(self.data_id))
                print("base_approach_dir: {}".format(gen_.base_approach_dir))
                print("sampling_dirs: {}".format(gen_.sampling_dirs))
                print("sampling mode: {}".format(gen_.sampling_mode))
                print("fkr_xyzdelta: {}".format(gen_.xyzdelta))
                print("fkr_angle_inc: {}".format(gen_.angle_inc))
                print("fkr_angle_offset: {}".format(gen_.angle_offset))
                print("l1_from_ground: {}m".format(gen_.l1_from_ground))

                if gen_.base_approach_dir is None:
                    gen_.base_approach_dir = [1., 0]
                if gen_.sampling_dirs is None:
                    raise ValueError("Please set sampling_dirs")

                self.build_fkr(self.data_id, gen_.base_approach_dir, gen_.sampling_dirs, \
                               maxradius=gen_.max_radius, fkr_xyzdelta=gen_.xyzdelta, \
                               fkr_angle_inc=gen_.angle_inc, fkr_angle_offset=gen_.angle_offset, \
                               sampling_mode=gen_.sampling_mode, l1_from_ground=gen_.l1_from_ground)
                exit()
        else:
            assert self.has_file_name(self.data_id), "Database id [{}] is not found."
            if not self.loadFKR_HDF5(self.data_id):
                raise ValueError("Failed to load FKR hdf5 with id {}".format(self.data_id))
            print("base_approach_dir: {}".format(self.base_approach_dir))
            print("sampling_dirs: {}".format(self.fkr_sampling_dirs))

    def get_fkr_version(self):
        return self.fkr_version

    def focused_sampling(self, maxradius, delta, mode="visible-front", l1_from_ground=0.5175):
        nsteps = np.ceil(maxradius/delta)
        groundsteps = np.ceil(l1_from_ground/delta)
        modes = {
            # +ve X and Z axis only
            "visible-front": (0, nsteps, -nsteps, nsteps, -groundsteps, nsteps),
            # +ve Z axis
            "half-top": (-nsteps, nsteps, -nsteps, nsteps, 0, nsteps),
            # +ve X axis
            "half-front": (0, nsteps, -nsteps, nsteps, -nsteps, nsteps),
        }
        if mode not in modes.keys():
            raise ValueError("Invalid mode. Available modes: {}".format(modes.keys()))
        grid_extent = modes[mode]
        min_x, max_x, min_y, max_y, min_z, max_z = grid_extent
        X, Y, Z = np.mgrid[min_x:max_x, min_y:max_y, min_z:max_z]
        allpoints = np.c_[X.flat, Y.flat, Z.flat] * delta
        insideinds = np.flatnonzero(np.sum(allpoints**2, 1) < maxradius**2)
        return allpoints, insideinds, X.shape, np.array((1.0/delta, nsteps)), grid_extent

    def fkr_pcg(self, sampling_dirs, maxradius=None, fkr_xyzdelta=0.04, fkr_angle_inc=np.pi/6, 
                fkr_angle_offset=0., sampling_mode="visible-front", l1_from_ground=0.5175):
        """
        fkr_angle_inc: Rotation increment when generating 6D pose for each ray direction in radian
        fkr_angle_offset: Angle offset before applying angle increment
        sampling_mode: Defines volume to sample
        """
        self.fkr_xyzdelta = fkr_xyzdelta
        self.fkr_angle_inc = fkr_angle_inc
        self.fkr_angle_offset = fkr_angle_offset
        if not self.ikmodel.load():
            self.ikmodel.autogenerate()
        with self.robot:
            Tbase = self.manip.GetBase().GetTransform()
            Tbaseinv = np.linalg.inv(Tbase)
            Trobot = np.dot(Tbaseinv, self.robot.GetTransform())
            self.robot.SetTransform(Trobot) 
            maniplinks = self.getManipulatorLinks(self.manip)
            for link in self.robot.GetLinks():
                link.Enable(link in maniplinks)
            armjoints = self.getOrderedArmJoints()
            baseanchor = armjoints[0].GetAnchor()
            eetrans = self.manip.GetEndEffectorTransform()[0:3, 3]
            armlength = 0
            for j in armjoints[::-1]:
                armlength += np.sqrt(np.sum((eetrans-j.GetAnchor())**2))
                eetrans = j.GetAnchor()
            if maxradius is None:
                maxradius = armlength+fkr_xyzdelta*np.sqrt(3.0)*1.05

        allpoints, insideinds, shape, self.fkr_pointscale, self.fkr_extent = \
                    self.focused_sampling(maxradius, fkr_xyzdelta, sampling_mode, l1_from_ground)
        """
        Each voxel is sampled from all sampling_dirs.
        We sample the reachability of one sampling_dir by discretizing
        the last joint angles by fkr_angle_inc. Thus, each sample_dir
        will be sampled from X number of discretized joint angles where
        Y number of them will be reachable.
        - fkr_qdensity3d: average no of IK solutions per sampling_dir
        - fkr_rotdensity3d: average reachable angle ratio (Y/X) per sampling_dir
        - fkr_3d: indicates whether all the sampling_dirs of the voxel has
                  at least one IK solution
        """
        self.fkr_qdensity3d = np.zeros(np.prod(shape))
        self.fkr_rotdensity3d = np.zeros(np.prod(shape))
        self.fkr_3d = np.zeros(np.prod(shape))

        def producer():
            T = np.eye(4)
            for i, ind in enumerate(insideinds):
                T[:3, 3] = allpoints[ind]+baseanchor
                if np.mod(i, 1000) == 0:
                    print("FKR {}/{}".format(i, len(insideinds)))
                yield ind, T

        def consumer(ind, T):
            with self.robot:
                self.robot.SetTransform(Trobot)
                numvalid = 0
                numrotvalid = 0
                for ray_dir in sampling_dirs:
                    num_valid_angle_per_ray = 0
                    found_one_ray_solution = False
                    Tray = baldor.transform.between_axes(baldor.Z_AXIS, ray_dir)
                    angle_count = 0
                    for angle in np.arange(0, 2*np.pi, fkr_angle_inc):
                        angle_count += 1
                        Toffset = orpy.matrixFromAxisAngle((fkr_angle_offset + angle)*baldor.Z_AXIS)
                        T[:3, :3] = np.dot(Tray, Toffset)[:3, :3]
                        solutions = self.manip.FindIKSolutions(T, 0)
                        if solutions is not None:
                            if len(solutions) > 0:
                                numvalid += len(solutions)
                                num_valid_angle_per_ray += 1
                                found_one_ray_solution = True
                    if not found_one_ray_solution:
                        return ind, 0, 0, 0
                    numrotvalid += num_valid_angle_per_ray / angle_count
                return ind, numvalid, numrotvalid, 1

        def gatherer(ind=None, numvalid=None, numrotvalid=None, valid=None):
            if ind is not None:
                self.fkr_qdensity3d[ind] = numvalid/float(len(sampling_dirs))
                self.fkr_rotdensity3d[ind] = numrotvalid/float(len(sampling_dirs))
                self.fkr_3d[ind] = valid
            else:
                self.fkr_qdensity3d = np.reshape(self.fkr_qdensity3d, shape)
                self.fkr_rotdensity3d = np.reshape(self.fkr_rotdensity3d, shape)
                self.fkr_3d = np.reshape(self.fkr_3d, shape)

        return producer, consumer, gatherer, len(insideinds)

    def build_fkr(self, database_id, base_approach_dir, sampling_dirs, 
                                             maxradius=None, fkr_xyzdelta=0.04, 
                                             fkr_angle_inc=np.pi/6, fkr_angle_offset=0., 
                                             sampling_mode="visible-front", l1_from_ground=0.5175):
        self.base_approach_dir = base_approach_dir
        self.fkr_sampling_dirs = sampling_dirs

        starttime = time.time()
        producer, consumer, gatherer, numjobs = self.fkr_pcg(sampling_dirs, maxradius, fkr_xyzdelta, \
                                                             fkr_angle_inc, fkr_angle_offset, \
                                                             sampling_mode, l1_from_ground)
        print("FKR database has {} items.".format(numjobs))
        for work in producer():
            results = consumer(*work)
            if len(results) > 0:
                gatherer(*results)
        gatherer()
        self.fkr_build_time = time.time()-starttime
        print("FKR database finished in {}".format(self.fkr_build_time))
        self.saveFKR_HDF5(database_id)

    def get_file_name(self, database_id=None, read=False):
        return orpy.RaveFindDatabaseFile(os.path.join('robot.'+self.robot.GetKinematicsGeometryHash(), \
                                                      'fkr.'+database_id+self.manip.GetStructureHash()+'.pp'), read)

    def has_file_name(self, database_id):
        return os.path.isfile(self.get_file_name(database_id=database_id))

    def saveFKR_HDF5(self, database_id):
        filename = self.get_file_name(database_id, False)
        try:
            os.makedirs(os.path.split(filename)[0])
        except OSError:
            pass
        with h5py.File(filename, 'w') as f:
            f['version'] = self.get_fkr_version()
            f['3d'] = self.fkr_3d
            f['qdensity3d'] = self.fkr_qdensity3d
            f['rotdensity3d'] = self.fkr_rotdensity3d
            f['extent'] = self.fkr_extent
            f['pointscale'] = self.fkr_pointscale
            f['xyzdelta'] = self.fkr_xyzdelta
            f['angle_inc'] = self.fkr_angle_inc
            f['angle_offset'] = self.fkr_angle_offset
            f['sampling_dirs'] = self.fkr_sampling_dirs
            f['base_approach_dir'] = self.base_approach_dir
            f['build_time'] = self.fkr_build_time
        print("Saved model to {}".format(filename))

    def _Close_FKR_Database(self):
        if self._fkr_databasefile is not None:
            try:
                if self._fkr_databasefile:
                    self._fkr_databasefile.close()
            except Exception as e:
                print(e)
            self._fkr_databasefile = None

    def has_fkr(self):
        return self.fkr_3d is not None and \
               self.fkr_qdensity3d is not None and \
               self.fkr_rotdensity3d is not None

    def loadFKR_HDF5(self, database_id):
        filename = self.get_file_name(database_id, True)
        if len(filename) == 0:
            return False

        self._Close_FKR_Database()
        try:
            f = h5py.File(filename, 'r')
            if f['version'].value != self.get_fkr_version():
                #log.error('version is wrong %s!=%s ',f['fkr_version'],self.get_fkr_version())
                print("Version is wrong {}!={}".format(f['version'].value, self.get_fkr_version()))
                return False

            self.fkr_3d = f['3d'].value
            self.fkr_qdensity3d = f['qdensity3d'].value
            self.fkr_rotdensity3d = f['rotdensity3d'].value
            self.fkr_extent = f['extent'].value
            self.fkr_pointscale = f['pointscale'].value
            self.fkr_xyzdelta = f['xyzdelta'].value
            self.fkr_angle_inc = f['angle_inc'].value
            self.fkr_angle_offset = f['angle_offset'].value
            self.fkr_sampling_dirs = f['sampling_dirs'].value
            self.base_approach_dir = f['base_approach_dir'].value
            self.fkr_build_time = f['build_time'].value
            self._fkr_databasefile = f
            f = None
            print("Loaded FKR HDF5 with id: {}".format(database_id))
            return self.has_fkr()

        except Exception as e:
            print("Fail loading FKR HDF5 for {}: {}".format(filename, e))
            return False
        finally:
            if f is not None:
                f.close()

    def visualize(self, j1_offset=None, l0_name='link0', l1_name='link1', 
                        showlimits=True, reach_param=None, showrobot=True, showori=True, 
                        valid_fkr_color=(0, 1, 0), valid_fkr_opacity=0.3,
                        azimuth=None, elevation=None, distance=None, focalpoint=None):
        from mayavi import mlab
        mlab.figure("fkr", fgcolor=(0, 0, 0), bgcolor=(1, 1, 1), size=(768, 768))
        mlab.clf()

        # Get indices of valid fkr_3d
        vx, vy, vz = np.where(self.fkr_3d == 1.)
        # Get indices of nonvalid fkr_3d
        nx, ny, nz = np.where(self.fkr_3d < 1.)
        # Translate indices to mayavi grid indices
        min_x, max_x = self.fkr_extent[0:2]
        min_y, max_y = self.fkr_extent[2:4]
        min_z, max_z = self.fkr_extent[4:]
        vx -= int(np.abs(min_x))
        vy -= int(np.abs(min_y))
        vz -= int(np.abs(min_z))
        nx -= int(np.abs(min_x))
        ny -= int(np.abs(min_y))
        nz -= int(np.abs(min_z))

        # Show valid fkr
        if j1_offset is None:
            j1_offset = get_link_offset(self.robot, l0_name=l0_name, l1_name=l1_name)
        trans = voxel_unit_from_meter(np.array(j1_offset), self.fkr_xyzdelta)

        X, Y, Z = np.mgrid[min_x+trans[0]:max_x+trans[0],
                            min_y+trans[1]:max_y+trans[1],
                            min_z+trans[2]:max_z+trans[2]]
        fkr_src = mlab.pipeline.scalar_field(X, Y, Z, self.fkr_3d)
        iso_fkr = mlab.pipeline.iso_surface(fkr_src, contours=[1.0], color=valid_fkr_color,
                                            opacity=valid_fkr_opacity, vmin=0., vmax=1.)

        offset = np.array((0, 0, 0))
        if showrobot:
            with self.robot:
                Tbase = self.manip.GetBase().GetTransform()
                Tbaseinv = np.linalg.inv(Tbase)
                self.robot.SetTransform(np.dot(Tbaseinv, self.robot.GetTransform()))
                baseanchor = self.getOrderedArmJoints()[0].GetAnchor()
                trimesh = self.env.Triangulate(self.robot)
                v = trimesh.vertices/self.fkr_xyzdelta
                mlab.triangular_mesh(v[:, 0]-offset[0], v[:, 1]-offset[1], 
                                    v[:, 2]-offset[2], trimesh.indices, color=(0.5, 0.5, 0.5))

        # Display origin
        if showori:
            ori_x, ori_y, ori_z = display_mayavi_ori_axes()

        # Display limit planes & spheres
        if showlimits and reach_param is not None:
            # Display minimum/maximum Z plane & minimum X plane
            y_off = voxel_unit_from_meter(0.0, self.fkr_xyzdelta)
            yext = voxel_unit_from_meter(1.0, self.fkr_xyzdelta)
            if reach_param.Zmin_wrt_arm is not None:
                x_off = voxel_unit_from_meter(reach_param.Xmin_wrt_arm, self.fkr_xyzdelta)
                zmin_off = voxel_unit_from_meter(reach_param.Zmin_wrt_arm, self.fkr_xyzdelta)
                xext = voxel_unit_from_meter(1.0, self.fkr_xyzdelta)
                hpx, hpy, hpz = np.mgrid[x_off:x_off+xext:2j, y_off-yext:y_off+yext:2j, zmin_off:zmin_off:1j]
                hpx = np.reshape(hpx, (2, 2))
                hpy = np.reshape(hpy, (2, 2))
                hpz = np.reshape(hpz, (2, 2))
                zmin_plane = mlab.mesh(hpx, hpy, hpz, color=(0, 0, 1), opacity=0.3, name='zmin-plane')
            if reach_param.Zmax_wrt_arm is not None:
                x_off = voxel_unit_from_meter(reach_param.Xmin_wrt_arm, self.fkr_xyzdelta)
                zmax_off = voxel_unit_from_meter(reach_param.Zmax_wrt_arm, self.fkr_xyzdelta)
                xext = voxel_unit_from_meter(1.0, self.fkr_xyzdelta)
                hpx, hpy, hpz = np.mgrid[x_off:x_off+xext:2j, y_off-yext:y_off+yext:2j, zmax_off:zmax_off:1j]
                hpx = np.reshape(hpx, (2, 2))
                hpy = np.reshape(hpy, (2, 2))
                hpz = np.reshape(hpz, (2, 2))
                zmax_plane = mlab.mesh(hpx, hpy, hpz, color=(0, 0, 1), opacity=0.3, name='zmax-plane')
            if reach_param.Xmin_wrt_arm is not None:
                xmin_off = voxel_unit_from_meter(reach_param.Xmin_wrt_arm, self.fkr_xyzdelta)
                if reach_param.Zmax_wrt_arm is not None:
                    zext = zmax_off - zmin_off
                else:
                    zext = voxel_unit_from_meter(1.5, self.fkr_xyzdelta)
                hpx,hpy,hpz = np.mgrid[xmin_off:xmin_off:1j, y_off-yext:y_off+yext:2j, zmin_off:zmin_off+zext:2j]
                hpx = np.reshape(hpx, (2, 2))
                hpy = np.reshape(hpy, (2, 2))
                hpz = np.reshape(hpz, (2, 2))
                xmin_plane = mlab.mesh(hpx, hpy, hpz, color=(0, 0, 1), opacity=0.3, name='zmin-plane')

            # Display limit spheres
            spheres_center_x = voxel_unit_from_meter(reach_param.spheres_center_wrt_arm[0], self.fkr_xyzdelta)
            spheres_center_y = voxel_unit_from_meter(reach_param.spheres_center_wrt_arm[1], self.fkr_xyzdelta)
            spheres_center_z = voxel_unit_from_meter(reach_param.spheres_center_wrt_arm[2], self.fkr_xyzdelta)
            rmin = voxel_unit_from_meter(reach_param.Rmin, self.fkr_xyzdelta)
            rmax = voxel_unit_from_meter(reach_param.Rmax, self.fkr_xyzdelta)
            spheres = []
            spheres += [mlab.points3d(spheres_center_x, spheres_center_y, spheres_center_z, 2*rmin, 
                                      scale_factor=1, color=(1, 0, 0),
                                      resolution=10, opacity=0.2, name='inner-sphere')]
            spheres += [mlab.points3d(spheres_center_x, spheres_center_y, spheres_center_z, 2*rmax, 
                                      scale_factor=1, color=(1, 0, 0),
                                      resolution=10, opacity=0.2, name='outer-sphere')]
        # Mayavi view
        v = mlab.view()
        if azimuth is None:
            azimuth = v[0]
        if elevation is None:
            elevation = v[1]
        if distance is None:
            distance = v[2]
        if focalpoint is None:
            focalpoint = v[3]
        display_mayavi_viewpoint([azimuth, elevation, distance, focalpoint])
        mlab.view(azimuth=azimuth, elevation=elevation, distance=distance, focalpoint=focalpoint)
        mlab.show()

    def calculate_reach_limits(self, Xmin_wrt_arm=0.25, Zmin_wrt_arm=0.3, Zmax_wrt_arm=None, \
                               arm_ori_wrt_base=[0.2115, 0., 0.320], safe_margin=0.05, \
                               l0_name='link0', l1_name='link1', l2_name='link2'):
        # Calculate position of spheres' center
        if not self.ikmodel.load():
            self.ikmodel.autogenerate()
        with self.robot:
            Tbase = self.manip.GetBase().GetTransform()
            Tbaseinv = np.linalg.inv(Tbase)
            Trobot = np.dot(Tbaseinv, self.robot.GetTransform())
            self.robot.SetTransform(Trobot)
            # maniplinks = self.getManipulatorLinks(self.manip)
            # armjoints = self.getOrderedArmJoints()
            # eetrans = self.manip.GetEndEffectorTransform()[0:3, 3]
            # ee_length_wrt_j5 = np.sqrt(np.sum((eetrans-armjoints[-2].GetAnchor())**2))
            rot_center_wrt_arm = [0, 0, get_link_offset(self.robot, l0_name, l2_name)[2]]
        spheres_center_wrt_arm = rot_center_wrt_arm
        # Calculate position relative to robot's link 1 because FKR was calculated with origin at link 1
        l1_wrt_arm = get_link_offset(self.robot, l0_name, l1_name)
        spheres_center_wrt_l1 = spheres_center_wrt_arm - l1_wrt_arm

        # Get indices of valid fkr_3d
        vx, vy, vz = np.where(self.fkr_3d == 1.)
        min_x, max_x = self.fkr_extent[0:2]
        min_y, max_y = self.fkr_extent[2:4]
        min_z, max_z = self.fkr_extent[4:]
        vx -= int(np.abs(min_x))
        vy -= int(np.abs(min_y))
        vz -= int(np.abs(min_z))
        # Only consider valid voxels in region (x > Xmin, z > Zmin)
        xmin_voxel = int(voxel_unit_from_meter(Xmin_wrt_arm - l1_wrt_arm[0], self.fkr_xyzdelta))
        zmin_voxel = int(voxel_unit_from_meter(Zmin_wrt_arm - l1_wrt_arm[2], self.fkr_xyzdelta))
        if Zmax_wrt_arm is not None:
            zmax_voxel = int(voxel_unit_from_meter(Zmax_wrt_arm - l1_wrt_arm[2], self.fkr_xyzdelta))
        else:
            zmax_voxel = max(vz)+1

        def common(lst1, lst2): 
            return list(set(lst1) & set(lst2))

        # Get outer surface
        outer_xyzr = []
        for z_now in range(zmin_voxel, zmax_voxel+1, 1):
            z_indices = list(np.where(vz==z_now)[0])
            for x_now in range(xmin_voxel, max(vx)+1, 1):
                x_indices = list(np.where(vx==x_now)[0])
                y_indices = common(z_indices, x_indices)
                if len(y_indices) > 0:
                    vy_now = vy[y_indices]
                    # right space w.r.t. spheres' center
                    y_out = max(vy_now)
                    y_out_ind = y_indices[list(vy_now).index(y_out)]
                    y_out = meter_from_voxel_unit(y_out, self.fkr_xyzdelta)
                    x_out = meter_from_voxel_unit(vx[y_out_ind], self.fkr_xyzdelta)
                    z_out = meter_from_voxel_unit(vz[y_out_ind], self.fkr_xyzdelta)
                    r = np.linalg.norm(np.array([x_out, y_out, z_out])-np.array(spheres_center_wrt_l1))
                    outer_xyzr.append([x_out, y_out, z_out, r])
                    # left space w.r.t. spheres' center
                    y_out = min(vy_now)
                    y_out_ind = y_indices[list(vy_now).index(y_out)]
                    y_out = meter_from_voxel_unit(y_out, self.fkr_xyzdelta)
                    x_out = meter_from_voxel_unit(vx[y_out_ind], self.fkr_xyzdelta)
                    z_out = meter_from_voxel_unit(vz[y_out_ind], self.fkr_xyzdelta)
                    r = np.linalg.norm(np.array([x_out, y_out, z_out])-np.array(spheres_center_wrt_l1))
                    outer_xyzr.append([x_out, y_out, z_out, r])        
        # Calculate outer sphere' radius Rmax
        Rmax = min(np.array(outer_xyzr)[:,3])*(1-safe_margin)
        print("Rmax = {}".format(Rmax))

        # Get inner surface
        inner_xyzr = []
        for z_now in range(zmin_voxel, zmax_voxel+1, 1):
            z_indices = list(np.where(vz==z_now)[0])
            for x_now in range(xmin_voxel, max(vx)+1, 1):
                x_indices = list(np.where(vx==x_now)[0])
                y_indices = common(z_indices, x_indices)
                if len(y_indices) > 0:
                    vy_now = vy[y_indices]
                    # right space w.r.t. spheres' center
                    y_right = [i for i in vy_now if i >= spheres_center_wrt_l1[1]]
                    if len(y_right) > 0:
                        y_in = min(y_right)
                        if y_in >= spheres_center_wrt_l1[1]+1:
                            y_in_ind = y_indices[list(vy_now).index(y_in)]
                            y_in = meter_from_voxel_unit(y_in, self.fkr_xyzdelta)
                            x_in = meter_from_voxel_unit(vx[y_in_ind], self.fkr_xyzdelta)
                            z_in = meter_from_voxel_unit(vz[y_in_ind], self.fkr_xyzdelta)
                            r = np.linalg.norm(np.array([x_in, y_in, z_in])-np.array(spheres_center_wrt_l1))
                            inner_xyzr.append([x_in, y_in, z_in, r])
                    # right space w.r.t. spheres' center
                    y_left = [i for i in vy_now if i < spheres_center_wrt_l1[1]]
                    if len(y_left) > 0:
                        y_in = max(y_left)
                        if y_in < spheres_center_wrt_l1[1]-1:
                            y_in_ind = y_indices[list(vy_now).index(y_in)]
                            y_in = meter_from_voxel_unit(y_in, self.fkr_xyzdelta)
                            x_in = meter_from_voxel_unit(vx[y_in_ind], self.fkr_xyzdelta)
                            z_in = meter_from_voxel_unit(vz[y_in_ind], self.fkr_xyzdelta)
                            r = np.linalg.norm(np.array([x_in, y_in, z_in])-np.array(spheres_center_wrt_l1))
                            inner_xyzr.append([x_in, y_in, z_in, r])
        # Calculate inner sphere' radius Rmax
        if len(inner_xyzr) > 0:
            Rmin = max(np.array(inner_xyzr)[:,3])*(1+safe_margin)
        else: 
            Rmin = 0
        print("Rmin = {}".format(Rmin))

        reach_param = ReachLimitParameters(Rmin, Rmax, Xmin_wrt_arm, Zmin_wrt_arm, Zmax_wrt_arm, \
                                           spheres_center_wrt_arm, arm_ori_wrt_base)
        return reach_param
# END