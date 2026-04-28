import { defineConfig } from 'vite'
import { cpSync, mkdirSync } from 'fs'
import { resolve } from 'path'

export default defineConfig({
  plugins: [
    {
      name: 'copy-mediapipe-wasm',
      closeBundle() {
        const src = resolve('node_modules/@mediapipe/tasks-vision/wasm')
        const dest = resolve('dist/wasm')
        mkdirSync(dest, { recursive: true })
        cpSync(src, dest, { recursive: true })
        console.log('MediaPipe WASM copied to dist/wasm')
      }
    }
  ]
})
