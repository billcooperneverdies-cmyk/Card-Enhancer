import { motion } from 'framer-motion';
import { 
  Sparkles, Wand2, Palette, Contrast, 
  VolumeX, Maximize, Shield, Image as ImageIcon 
} from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import type { EnhancementSettings as SettingsType } from '../services/api';

interface EnhancementSettingsProps {
  settings: SettingsType;
  onSettingsChange: (settings: SettingsType) => void;
}

interface SettingSectionProps {
  title: string;
  icon: React.ReactNode;
  enabled: boolean;
  onToggle: (value: boolean) => void;
  children?: React.ReactNode;
  badge?: string;
}

function SettingSection({ title, icon, enabled, onToggle, children, badge }: SettingSectionProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass rounded-xl p-5 border ${
        enabled ? 'border-cyan-500/30' : 'border-gray-800'
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
            enabled ? 'bg-cyan-500/20 text-cyan-400' : 'bg-gray-800 text-gray-500'
          }`}>
            {icon}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <Label className="text-base font-medium">{title}</Label>
              {badge && (
                <Badge variant="outline" className="text-xs border-cyan-500/30 text-cyan-400">
                  {badge}
                </Badge>
              )}
            </div>
            <p className="text-xs text-gray-500">
              {enabled ? 'Enabled' : 'Disabled'}
            </p>
          </div>
        </div>
        <Switch
          checked={enabled}
          onCheckedChange={onToggle}
          className="data-[state=checked]:bg-cyan-500"
        />
      </div>
      
      <motion.div
        initial={false}
        animate={{ 
          height: enabled ? 'auto' : 0,
          opacity: enabled ? 1 : 0
        }}
        transition={{ duration: 0.2 }}
        className="overflow-hidden"
      >
        {children}
      </motion.div>
    </motion.div>
  );
}

interface SliderControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}

function SliderControl({ label, value, min, max, step, onChange, disabled }: SliderControlProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-sm text-gray-400">{label}</Label>
        <span className="text-sm font-mono text-cyan-400">{value.toFixed(2)}</span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([v]) => onChange(v)}
        disabled={disabled}
        className="w-full"
      />
    </div>
  );
}

export function EnhancementSettings({ settings, onSettingsChange }: EnhancementSettingsProps) {
  const updateSetting = <K extends keyof SettingsType>(
    key: K, 
    value: SettingsType[K]
  ) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  return (
    <div className="space-y-4 pb-20">
      {/* Blemish Removal */}
      <SettingSection
        title="Blemish Removal"
        icon={<Sparkles className="w-5 h-5" />}
        enabled={settings.blemish_removal}
        onToggle={(v) => updateSetting('blemish_removal', v)}
        badge="AI Powered"
      >
        <div className="pt-2 space-y-4">
          <SliderControl
            label="Detection Sensitivity"
            value={settings.blemish_sensitivity}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => updateSetting('blemish_sensitivity', v)}
          />
          <p className="text-xs text-gray-500">
            Higher sensitivity detects more defects but may affect card details
          </p>
        </div>
      </SettingSection>

      {/* Sharpening */}
      <SettingSection
        title="Sharpening"
        icon={<Wand2 className="w-5 h-5" />}
        enabled={settings.sharpening}
        onToggle={(v) => updateSetting('sharpening', v)}
      >
        <div className="pt-2">
          <SliderControl
            label="Sharpening Amount"
            value={settings.sharpening_amount}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => updateSetting('sharpening_amount', v)}
          />
        </div>
      </SettingSection>

      {/* Color Correction */}
      <SettingSection
        title="Color Correction"
        icon={<Palette className="w-5 h-5" />}
        enabled={settings.color_correction}
        onToggle={(v) => updateSetting('color_correction', v)}
      >
        <div className="pt-2 space-y-4">
          <SliderControl
            label="Color Temperature"
            value={settings.color_temperature}
            min={-1}
            max={1}
            step={0.1}
            onChange={(v) => updateSetting('color_temperature', v)}
          />
          <SliderControl
            label="Saturation"
            value={settings.saturation}
            min={0}
            max={2}
            step={0.1}
            onChange={(v) => updateSetting('saturation', v)}
          />
        </div>
      </SettingSection>

      {/* Contrast Enhancement */}
      <SettingSection
        title="Contrast Enhancement"
        icon={<Contrast className="w-5 h-5" />}
        enabled={settings.contrast_enhancement}
        onToggle={(v) => updateSetting('contrast_enhancement', v)}
      >
        <div className="pt-2">
          <SliderControl
            label="Contrast Amount"
            value={settings.contrast_amount}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => updateSetting('contrast_amount', v)}
          />
        </div>
      </SettingSection>

      {/* Noise Reduction */}
      <SettingSection
        title="Noise Reduction"
        icon={<VolumeX className="w-5 h-5" />}
        enabled={settings.noise_reduction}
        onToggle={(v) => updateSetting('noise_reduction', v)}
      >
        <div className="pt-2">
          <SliderControl
            label="Noise Reduction Strength"
            value={settings.noise_reduction_strength}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => updateSetting('noise_reduction_strength', v)}
          />
        </div>
      </SettingSection>

      {/* Upscaling */}
      <SettingSection
        title="AI Upscaling"
        icon={<Maximize className="w-5 h-5" />}
        enabled={settings.upscaling}
        onToggle={(v) => updateSetting('upscaling', v)}
        badge="Slow"
      >
        <div className="pt-2">
          <div className="space-y-2">
            <Label className="text-sm text-gray-400">Upscale Factor</Label>
            <div className="flex gap-2">
              {[1, 2, 4].map((factor) => (
                <button
                  key={factor}
                  onClick={() => updateSetting('upscale_factor', factor)}
                  className={`px-4 py-2 rounded-lg border transition-all ${
                    settings.upscale_factor === factor
                      ? 'border-cyan-500 bg-cyan-500/20 text-cyan-400'
                      : 'border-gray-700 hover:border-gray-600'
                  }`}
                >
                  {factor}x
                </button>
              ))}
            </div>
          </div>
        </div>
      </SettingSection>

      <Separator className="bg-cyan-500/20" />

      {/* Output Settings */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-xl p-5 border border-gray-800"
      >
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-lg bg-gray-800 flex items-center justify-center">
            <ImageIcon className="w-5 h-5 text-gray-400" />
          </div>
          <Label className="text-base font-medium">Output Settings</Label>
        </div>

        <div className="space-y-4">
          {/* Preserve Holographic */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-cyan-400" />
              <Label className="text-sm">Preserve Holographic Regions</Label>
            </div>
            <Switch
              checked={settings.preserve_holographic}
              onCheckedChange={(v) => updateSetting('preserve_holographic', v)}
              className="data-[state=checked]:bg-cyan-500"
            />
          </div>

          {/* Output Format */}
          <div className="space-y-2">
            <Label className="text-sm text-gray-400">Output Format</Label>
            <div className="flex gap-2">
              {['png', 'jpg', 'webp', 'tiff'].map((format) => (
                <button
                  key={format}
                  onClick={() => updateSetting('output_format', format as 'png' | 'jpg' | 'webp' | 'tiff')}
                  className={`px-4 py-2 rounded-lg border uppercase text-sm transition-all ${
                    settings.output_format === format
                      ? 'border-cyan-500 bg-cyan-500/20 text-cyan-400'
                      : 'border-gray-700 hover:border-gray-600'
                  }`}
                >
                  {format}
                </button>
              ))}
            </div>
          </div>

          {/* Output Quality */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm text-gray-400">Output Quality</Label>
              <span className="text-sm font-mono text-cyan-400">{settings.output_quality}%</span>
            </div>
            <Slider
              value={[settings.output_quality]}
              min={50}
              max={100}
              step={5}
              onValueChange={([v]) => updateSetting('output_quality', v)}
              className="w-full"
            />
          </div>

          {/* Output DPI */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm text-gray-400">Output DPI</Label>
              <span className="text-sm font-mono text-cyan-400">{settings.output_dpi}</span>
            </div>
            <div className="flex gap-2">
              {[72, 150, 300, 600].map((dpi) => (
                <button
                  key={dpi}
                  onClick={() => updateSetting('output_dpi', dpi)}
                  className={`px-3 py-1.5 rounded-lg border text-sm transition-all ${
                    settings.output_dpi === dpi
                      ? 'border-cyan-500 bg-cyan-500/20 text-cyan-400'
                      : 'border-gray-700 hover:border-gray-600'
                  }`}
                >
                  {dpi}
                </button>
              ))}
            </div>
            <p className="text-xs text-gray-500">
              Higher DPI for print, lower for web/screen viewing
            </p>
          </div>
        </div>
      </motion.div>

      {/* Preset Buttons */}
      <div className="flex gap-2">
        <button
          onClick={() => onSettingsChange({
            ...settings,
            blemish_removal: true,
            blemish_sensitivity: 0.8,
            sharpening: true,
            sharpening_amount: 0.6,
            color_correction: true,
            contrast_enhancement: true,
            noise_reduction: true,
            upscaling: false,
            output_dpi: 300
          })}
          className="flex-1 px-4 py-2 rounded-lg border border-cyan-500/30 hover:bg-cyan-500/10 text-sm transition-all"
        >
          Restoration Preset
        </button>
        <button
          onClick={() => onSettingsChange({
            ...settings,
            blemish_removal: false,
            sharpening: true,
            sharpening_amount: 0.8,
            color_correction: false,
            contrast_enhancement: true,
            contrast_amount: 0.5,
            noise_reduction: false,
            upscaling: true,
            upscale_factor: 2,
            output_dpi: 300
          })}
          className="flex-1 px-4 py-2 rounded-lg border border-purple-500/30 hover:bg-purple-500/10 text-sm transition-all"
        >
          Upscale Preset
        </button>
        <button
          onClick={() => onSettingsChange({
            blemish_removal: true,
            blemish_sensitivity: 0.7,
            sharpening: true,
            sharpening_amount: 0.5,
            color_correction: true,
            color_temperature: 0,
            saturation: 1,
            contrast_enhancement: true,
            contrast_amount: 0.3,
            noise_reduction: true,
            noise_reduction_strength: 0.5,
            upscaling: false,
            upscale_factor: 2,
            preserve_holographic: true,
            output_format: 'png',
            output_quality: 95,
            output_dpi: 300
          })}
          className="px-4 py-2 rounded-lg border border-gray-700 hover:border-gray-600 text-sm transition-all"
        >
          Reset
        </button>
      </div>
    </div>
  );
}
