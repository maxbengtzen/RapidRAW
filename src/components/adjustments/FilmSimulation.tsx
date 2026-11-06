import React, { memo, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import Slider, { SliderChangeEvent } from '../ui/Slider';
import Switch from '../ui/Switch';
import Dropdown from '../ui/Dropdown';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';
import {
  Adjustments,
  FilmSimSettings,
  FilmStock,
  INITIAL_ADJUSTMENTS,
  areSectionPropsEqual,
} from '../../utils/adjustments';

interface FilmSimulationProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Partial<Adjustments>) => Partial<Adjustments>)): void;
  onDragStateChange?: (isDragging: boolean) => void;
}

const SPEKTRAFILM_URL = 'https://github.com/andreavolpato/spektrafilm';
const SPEKTRAFILM_NAME = 'spektrafilm';
const SPEKTRAFILM_ATTRIBUTION = '· Andrea Volpato · CC BY-SA 4.0';

let stocksPromise: Promise<FilmStock[]> | null = null;
const loadStocks = () => {
  if (!stocksPromise) {
    stocksPromise = invoke<FilmStock[]>('list_film_stocks').catch((e) => {
      console.error('Failed to list film stocks', e);
      stocksPromise = null;
      return [];
    });
  }
  return stocksPromise;
};

const AUTO_PAPER = '__auto__';

const METERING_METHODS = [
  'center_weighted',
  'average',
  'median',
  'partial',
  'matrix',
  'multi_zone',
  'highlight_weighted',
] as const;

const DIFFUSION_FILTERS = ['none', 'black_pro_mist', 'pro_mist', 'glimmerglass', 'cinebloom'] as const;

const Group = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <div className="space-y-3">
    <Text variant={TextVariants.label} className="block uppercase tracking-wide opacity-70">
      {title}
    </Text>
    {children}
  </div>
);

function FilmSimulationPanel({ adjustments, setAdjustments, onDragStateChange }: FilmSimulationProps) {
  const { t } = useTranslation();
  const [stocks, setStocks] = useState<FilmStock[]>([]);

  useEffect(() => {
    let alive = true;
    loadStocks().then((list) => alive && setStocks(list));
    return () => {
      alive = false;
    };
  }, []);

  const film: FilmSimSettings = { ...INITIAL_ADJUSTMENTS.filmSim, ...(adjustments.filmSim ?? {}) };
  const isActive = adjustments.toneMapper === 'film';
  const isMetered = film.autoExposure || film.matchCameraBrightness;

  const update = (changes: Partial<FilmSimSettings>) =>
    setAdjustments((prev: Partial<Adjustments>) => ({
      ...prev,
      toneMapper: 'film',
      filmSim: { ...INITIAL_ADJUSTMENTS.filmSim, ...(prev.filmSim ?? {}), ...changes },
    }));

  const stockOptions = useMemo(() => {
    const kindLabel = (s: FilmStock) =>
      s.isBw ? t('adjustments.film.kinds.bw') : t(`adjustments.film.kinds.${s.kind}`);
    return stocks
      .filter((s) => s.kind !== 'print')
      .sort((a, b) => Number(a.isBw) - Number(b.isBw) || a.kind.localeCompare(b.kind) || a.name.localeCompare(b.name))
      .map((s) => ({ label: `${s.name} · ${kindLabel(s)}`, value: s.id }));
  }, [stocks, t]);

  const selectedStock = stocks.find((s) => s.id === film.stock);
  const canPrint = !!selectedStock && selectedStock.kind === 'negative';

  const paperOptions = useMemo(() => {
    const target = selectedStock?.targetPrint;
    const targetName = stocks.find((s) => s.id === target)?.name;
    return [
      {
        label: targetName
          ? t('adjustments.film.paperAuto', { name: targetName })
          : t('adjustments.film.paperAutoPlain'),
        value: AUTO_PAPER,
      },
      ...stocks.filter((s) => s.kind === 'print').map((s) => ({ label: s.name, value: s.id })),
    ];
  }, [stocks, selectedStock, t]);

  const slider = (
    key: keyof FilmSimSettings,
    label: string,
    min: number,
    max: number,
    step: number,
    suffix?: string,
  ) => (
    <Slider
      label={label}
      min={min}
      max={max}
      step={step}
      suffix={suffix}
      value={film[key] as number}
      defaultValue={INITIAL_ADJUSTMENTS.filmSim[key] as number}
      onChange={(e: SliderChangeEvent) => update({ [key]: Number(e.target.value) })}
      onDragStateChange={onDragStateChange}
    />
  );

  return (
    <div className="space-y-4">
      <Switch
        label={t('adjustments.film.enable')}
        checked={isActive}
        onChange={(checked: boolean) =>
          setAdjustments((prev: Partial<Adjustments>) => ({
            ...prev,
            toneMapper: checked ? 'film' : 'basic',
            filmSim: { ...INITIAL_ADJUSTMENTS.filmSim, ...(prev.filmSim ?? {}) },
          }))
        }
      />

      <div className={isActive ? 'space-y-4' : 'space-y-4 opacity-50'}>
        <Group title={t('adjustments.film.groups.exposure')}>
          <Switch
            label={t('adjustments.film.matchCameraBrightness')}
            checked={film.matchCameraBrightness}
            onChange={(checked: boolean) => update({ matchCameraBrightness: checked })}
          />
          {film.matchCameraBrightness ? (
            <Text variant={TextVariants.small} className="block opacity-70">
              {t('adjustments.film.matchCameraBrightnessHint')}
            </Text>
          ) : (
            <Switch
              label={t('adjustments.film.autoExposure')}
              checked={film.autoExposure}
              onChange={(checked: boolean) => update({ autoExposure: checked })}
            />
          )}
          {isMetered && (
            <Dropdown<string>
              options={METERING_METHODS.map((m) => ({ label: t(`adjustments.film.metering.${m}`), value: m }))}
              value={film.autoExposureMethod}
              onChange={(value) => update({ autoExposureMethod: value })}
            />
          )}
          {slider(
            'filmExposure',
            isMetered ? t('adjustments.film.exposureCompensation') : t('adjustments.film.filmExposure'),
            -3,
            3,
            0.05,
            ' EV',
          )}
        </Group>

        <Group title={t('adjustments.film.groups.camera')}>
          {slider('filmFormat', t('adjustments.film.filmFormat'), 4, 120, 1, ' mm')}
          <div>
            <Text variant={TextVariants.label} className="mb-2 block">
              {t('adjustments.film.diffusionFilter')}
            </Text>
            <Dropdown<string>
              options={DIFFUSION_FILTERS.map((f) => ({ label: t(`adjustments.film.filters.${f}`), value: f }))}
              value={film.diffusionFilter || 'none'}
              onChange={(value) => update({ diffusionFilter: value })}
            />
          </div>
          {film.diffusionFilter !== 'none' &&
            slider('diffusionStrength', t('adjustments.film.diffusionStrength'), 0, 2, 0.05)}
        </Group>

        <Group title={t('adjustments.film.groups.film')}>
          <Dropdown<string>
            options={stockOptions}
            value={film.stock}
            onChange={(value) => update({ stock: value, paper: null, scanFilm: false })}
          />
          {slider('grain', t('adjustments.film.grain'), 0, 2, 0.05)}
          {slider('halation', t('adjustments.film.halation'), 0, 3, 0.05)}
          {slider('couplers', t('adjustments.film.couplers'), 0, 2, 0.05)}
        </Group>

        {canPrint && (
          <Group title={t('adjustments.film.groups.print')}>
            <Switch
              label={t('adjustments.film.scanFilm')}
              checked={film.scanFilm}
              onChange={(checked: boolean) => update({ scanFilm: checked })}
            />
            {!film.scanFilm && (
              <>
                <Dropdown<string>
                  options={paperOptions}
                  value={film.paper ?? AUTO_PAPER}
                  onChange={(value) => update({ paper: value === AUTO_PAPER ? null : value })}
                />
                {slider('printExposure', t('adjustments.film.printExposure'), -2, 2, 0.05, ' EV')}
                {slider('yellowShift', t('adjustments.film.yellowShift'), -30, 30, 1, ' cc')}
                {slider('magentaShift', t('adjustments.film.magentaShift'), -30, 30, 1, ' cc')}
              </>
            )}
          </Group>
        )}

        <Text variant={TextVariants.small} className="block opacity-70">
          {t('adjustments.film.credit')}{' '}
          <a href={SPEKTRAFILM_URL} target="_blank" rel="noreferrer" className="underline">
            {SPEKTRAFILM_NAME}
          </a>{' '}
          {SPEKTRAFILM_ATTRIBUTION}
        </Text>
      </div>
    </div>
  );
}

const areFilmSectionPropsEqual = areSectionPropsEqual('film');

// toneMapper is owned by the Basic section but drives this panel's switch.
export default memo(
  FilmSimulationPanel,
  (prev: FilmSimulationProps, next: FilmSimulationProps) =>
    prev.adjustments?.toneMapper === next.adjustments?.toneMapper &&
    areFilmSectionPropsEqual(prev as unknown as Record<string, unknown>, next as unknown as Record<string, unknown>),
);
