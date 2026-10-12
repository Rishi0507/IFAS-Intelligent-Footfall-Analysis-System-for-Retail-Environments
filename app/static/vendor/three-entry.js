// Entry for the vendored three.js bundle used by the IFAS dashboard.
// Only the pieces the dashboard draws with are re-exported, so the bundler can
// drop the rest. Rebuild with:
//   npx esbuild three-entry.js --bundle --minify --format=esm --outfile=three.bundle.min.js
export {
  ACESFilmicToneMapping, AmbientLight, BoxGeometry, BufferAttribute, BufferGeometry,
  CanvasTexture, CapsuleGeometry, CircleGeometry, Color, CylinderGeometry,
  DirectionalLight, DoubleSide, DynamicDrawUsage, Float32BufferAttribute, Group,
  HemisphereLight, InstancedMesh, Line, LineBasicMaterial, LineSegments, LinearFilter,
  MathUtils, Matrix4, Mesh, MeshBasicMaterial, MeshStandardMaterial, Object3D,
  PerspectiveCamera, PlaneGeometry, Quaternion, Raycaster, RingGeometry, Scene,
  SphereGeometry, Sprite, SpriteMaterial, SRGBColorSpace, TextureLoader, Vector2, Vector3,
  WebGLRenderer, REVISION,
} from 'three';
export { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
export { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
