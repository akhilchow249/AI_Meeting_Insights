import React, { useState, useCallback } from 'react'
import Navbar from './components/Navbar'
import UploadScreen from './screens/Upload'
import PipelineScreen from './screens/Pipeline'
import ReportScreen from './screens/Report'
import LibraryScreen from './screens/Library'
import ObservabilityScreen from './screens/Observability'

export default function App() {
  const [screen, setScreen] = useState('upload')
  const [sessionId, setSessionId] = useState(null)
  const [uploadMeta, setUploadMeta] = useState(null)

  const handleProcessStart = useCallback((sid, meta) => {
    setSessionId(sid)
    setUploadMeta(meta)
    setScreen('pipeline')
  }, [])

  const handlePipelineComplete = useCallback((sid) => {
    setSessionId(sid)
    setScreen('report')
  }, [])

  const handleOpenReport = useCallback((sid) => {
    setSessionId(sid)
    setScreen('report')
  }, [])

  const handleNav = useCallback((target) => {
    setScreen(target)
  }, [])

  return (
    <div className="bg-bg text-ink min-h-screen font-body">
      <Navbar
        screen={screen}
        onNav={handleNav}
        sessionId={sessionId}
      />

      {screen === 'upload' && (
        <UploadScreen onProcessStart={handleProcessStart} />
      )}

      {screen === 'pipeline' && (
        <PipelineScreen
          sessionId={sessionId}
          onComplete={handlePipelineComplete}
        />
      )}

      {screen === 'report' && (
        <ReportScreen sessionId={sessionId} />
      )}

      {screen === 'library' && (
        <LibraryScreen onOpenReport={handleOpenReport} />
      )}

      {screen === 'observability' && (
        <ObservabilityScreen />
      )}
    </div>
  )
}
