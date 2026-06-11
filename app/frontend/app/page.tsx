'use client'

import { useRouter } from 'next/navigation'
import styles from './landing.module.css'
import ThemeToggle from '../components/ThemeToggle'

export default function Home() {
  const router = useRouter()

  return (
    <div className={styles.container}>
      <div className={styles.themeToggleWrapper}>
        <ThemeToggle />
      </div>
      
      <div className={styles.hero}>
        <div className={styles.heroGrid}></div>
        <h1>mllabiome-ii</h1>
        <p className={styles.subtitle}>interactive inference for packaged microbiome models</p>
        <p className={styles.description}>Upload a sample table, choose a hosted microbiome model, and inspect prediction-level feature attributions.</p>
      </div>

      <div className={styles.modeSelector}>
        <div className={styles.modeCard} onClick={() => router.push('/inference')}>
          <div className={styles.cardHeader}>
            <h2>Model inference</h2>
          </div>
          <p>Serve exported model packages directly from <code>app/backend/production_models</code>.</p>
          <button className={styles.modeButton}>Open Inference</button>
        </div>
      </div>

      <div className={styles.info}>
        <h3>Interactive inference flow</h3>
        <div className={styles.steps}>
          <div className={styles.step}>
            <span className={styles.stepNumber}>01</span>
            <div>
              <h4>Drop in model package</h4>
              <p>Copy an exported model zip into the backend production model directory.</p>
            </div>
          </div>
          <div className={styles.step}>
            <span className={styles.stepNumber}>02</span>
            <div>
              <h4>Upload sample profile</h4>
              <p>Use wide sample-by-feature TSV/CSV or MetaPhlAn-style profile tables.</p>
            </div>
          </div>
          <div className={styles.step}>
            <span className={styles.stepNumber}>03</span>
            <div>
              <h4>Run packaged inference</h4>
              <p>Predictions are computed from the selected hosted model package.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
