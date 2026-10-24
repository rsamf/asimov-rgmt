// three.js renderer for a MuJoCo model: one Mesh per visual geom (group 2),
// posed every frame from geom_xpos / geom_xmat. MuJoCo is Z-up; the whole
// scene lives under a group rotated so that MuJoCo (x, y, z) -> three (x, z, -y).
import * as THREE from 'three';

const VISUAL_GROUP = 2;
const mjToThree = (x, y, z) => new THREE.Vector3(x, z, -y);

function gridTexture() {
  const s = 512, c = document.createElement('canvas');
  c.width = c.height = s;
  const g = c.getContext('2d');
  g.fillStyle = '#d9dde3'; g.fillRect(0, 0, s, s);
  g.fillStyle = '#ced3da'; g.fillRect(0, 0, s / 2, s / 2); g.fillRect(s / 2, s / 2, s / 2, s / 2);
  g.strokeStyle = '#9aa3ae'; g.lineWidth = 3; g.strokeRect(0, 0, s, s);
  g.strokeStyle = '#b7bec7'; g.lineWidth = 1.5;
  for (let i = 1; i < 4; i++) {
    g.beginPath(); g.moveTo(i * s / 4, 0); g.lineTo(i * s / 4, s); g.stroke();
    g.beginPath(); g.moveTo(0, i * s / 4); g.lineTo(s, i * s / 4); g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  return t;
}

function meshGeometry(model, meshId) {
  const vadr = model.mesh_vertadr[meshId];
  const fadr = model.mesh_faceadr[meshId], fnum = model.mesh_facenum[meshId];
  const vert = model.mesh_vert, face = model.mesh_face;
  const pos = new Float32Array(fnum * 9);
  let nrm = null;
  try {
    const nadr = model.mesh_normaladr[meshId];
    const normal = model.mesh_normal, fn = model.mesh_facenormal;
    if (normal && fn && normal.length) {
      nrm = new Float32Array(fnum * 9);
      for (let f = 0; f < fnum; f++) for (let c = 0; c < 3; c++) {
        const ni = nadr + fn[3 * (fadr + f) + c];
        for (let k = 0; k < 3; k++) nrm[9 * f + 3 * c + k] = normal[3 * ni + k];
      }
    }
  } catch (e) { nrm = null; }
  for (let f = 0; f < fnum; f++) for (let c = 0; c < 3; c++) {
    const vi = vadr + face[3 * (fadr + f) + c];
    for (let k = 0; k < 3; k++) pos[9 * f + 3 * c + k] = vert[3 * vi + k];
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  if (nrm) geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  else geo.computeVertexNormals();
  return geo;
}

function primitiveGeometry(type, size) {
  // mjtGeom: 2 sphere, 3 capsule, 4 ellipsoid, 5 cylinder, 6 box
  switch (type) {
    case 2: return new THREE.SphereGeometry(size[0], 24, 16);
    case 3: { const g = new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 16); g.rotateX(Math.PI / 2); return g; }
    case 4: { const g = new THREE.SphereGeometry(1, 24, 16); g.scale(size[0], size[1], size[2]); return g; }
    case 5: { const g = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 24); g.rotateX(Math.PI / 2); return g; }
    case 6: return new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    default: return null;
  }
}

export class Renderer {
  constructor(container) {
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.setSize(container.clientWidth, container.clientHeight);
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    r.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(r.domElement);

    const scene = this.scene = new THREE.Scene();
    scene.background = new THREE.Color('#e8ecf1');
    scene.fog = new THREE.Fog('#e8ecf1', 12, 40);
    this.camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.05, 200);

    scene.add(new THREE.HemisphereLight('#ffffff', '#8a8f99', 1.1));
    const sun = this.sun = new THREE.DirectionalLight('#ffffff', 2.2);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const sc = sun.shadow.camera; sc.left = -3; sc.right = 3; sc.top = 3; sc.bottom = -3; sc.near = 0.5; sc.far = 20;
    sun.shadow.bias = -0.0005; sun.shadow.normalBias = 0.02;
    scene.add(sun, sun.target);

    const tex = gridTexture(); tex.repeat.set(200, 200);
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400),
      new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95, metalness: 0 }));
    ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true;
    scene.add(ground);

    this.world = new THREE.Group();           // MuJoCo frame
    this.world.rotation.x = -Math.PI / 2;
    scene.add(this.world);
    this.geoms = [];
    this.cam = { xy: null, yaw: 0, distance: 3.0, height: 1.6, targetHeight: 0.8, halflife: 0.25 };

    window.addEventListener('resize', () => {
      const w = container.clientWidth, h = container.clientHeight;
      r.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    });
  }

  buildModel(model) {
    const ngeom = model.ngeom;
    const group = model.geom_group, type = model.geom_type, dataid = model.geom_dataid;
    const matid = model.geom_matid, grgba = model.geom_rgba, mrgba = model.mat_rgba, size = model.geom_size;
    const cache = new Map();
    for (let g = 0; g < ngeom; g++) {
      if (group[g] !== VISUAL_GROUP) continue;
      let geo = null;
      if (type[g] === 7) {                      // mjGEOM_MESH
        const id = dataid[g];
        if (!cache.has(id)) cache.set(id, meshGeometry(model, id));
        geo = cache.get(id);
      } else {
        geo = primitiveGeometry(type[g], [size[3 * g], size[3 * g + 1], size[3 * g + 2]]);
      }
      if (!geo) continue;
      const m = matid[g];
      const rgba = m >= 0 ? [mrgba[4 * m], mrgba[4 * m + 1], mrgba[4 * m + 2], mrgba[4 * m + 3]]
                          : [grgba[4 * g], grgba[4 * g + 1], grgba[4 * g + 2], grgba[4 * g + 3]];
      const grey = Math.abs(rgba[0] - rgba[1]) < 0.02 && Math.abs(rgba[1] - rgba[2]) < 0.02;
      const color = grey ? new THREE.Color('#dfe3e9') : new THREE.Color(rgba[0], rgba[1], rgba[2]);
      const mat = new THREE.MeshStandardMaterial({
        color, roughness: 0.5, metalness: 0.1,
        transparent: rgba[3] < 1, opacity: rgba[3],
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true; mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      this.world.add(mesh);
      this.geoms.push({ g, mesh });
    }
    console.log(`[render] ${this.geoms.length} visual geoms, ${cache.size} meshes`);
  }

  update(data, rootPos, rootYaw, dt) {
    const xp = data.geom_xpos, xm = data.geom_xmat;
    for (const { g, mesh } of this.geoms) {
      const p = 3 * g, r = 9 * g;
      mesh.matrix.set(
        xm[r], xm[r + 1], xm[r + 2], xp[p],
        xm[r + 3], xm[r + 4], xm[r + 5], xp[p + 1],
        xm[r + 6], xm[r + 7], xm[r + 8], xp[p + 2],
        0, 0, 0, 1);
      mesh.matrixWorldNeedsUpdate = true;
    }
    // follow camera (locomotion/play.py FollowCamera)
    const c = this.cam;
    const [x, y] = rootPos;
    if (!c.xy) { c.xy = [x, y]; c.yaw = rootYaw; }
    else {
      const a = 1 - Math.pow(0.5, dt / c.halflife);
      c.xy[0] += (x - c.xy[0]) * a; c.xy[1] += (y - c.xy[1]) * a;
      let dy = rootYaw - c.yaw; dy = Math.atan2(Math.sin(dy), Math.cos(dy));
      c.yaw += dy * a;
    }
    const pos = mjToThree(c.xy[0] - c.distance * Math.cos(c.yaw), c.xy[1] - c.distance * Math.sin(c.yaw), c.height);
    const tgt = mjToThree(c.xy[0], c.xy[1], c.targetHeight);
    this.camera.position.copy(pos);
    this.camera.lookAt(tgt);
    // sun follows the robot so the shadow frustum stays tight
    const s = mjToThree(x, y, 0);
    this.sun.position.set(s.x + 2.5, 6, s.z + 3.5);
    this.sun.target.position.copy(s);
    this.renderer.render(this.scene, this.camera);
  }

  resetCamera() { this.cam.xy = null; }
}
